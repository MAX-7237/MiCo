"""CPU native32 contract tests for same-layer hybrid MiCo pruning."""
import copy
from fractions import Fraction
import importlib.util
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_test_utils import model, inputs, mask, GEN, torch
from unified_core import install_pruning
from budget_schedule import mico_budget


class LayerCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU tests must never initialize CUDA')

    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU tests unexpectedly initialized CUDA')

    def hybrid(self, m, target=8, cut=4):
        return install_pruning(m, 'mico', target, cut_after_layer=cut,
            quality_mode='native_hybrid_task_v1', record_s2_hidden_digest=True)

    def test_all_layer_budgets_and_rejected_alpha(self):
        for target in (128,256):
            for cut in range(1,16):
                result=mico_budget(target,784,cut_after_layer=cut)
                self.assertEqual(result['n1'],2*target)
                self.assertEqual(result['n2'],round(Fraction(32*target-cut*2*target,32-cut)))
        for cut in (0,16,True,4.0):
            with self.assertRaises(ValueError):mico_budget(128,784,cut_after_layer=cut)
        for pair in ((3,2),(5,4)):
            with self.assertRaises(ValueError):mico_budget(128,784,alpha_num=pair[0],alpha_den=pair[1])

    @torch.no_grad()
    def test_all_fifteen_sources_budgets_cache_and_logical_positions(self):
        data=inputs(12)
        text_count=data['input_ids'].shape[1]-36
        for cut in range(1,16):
            with self.subTest(cut=cut):
                m=model(); ctl=self.hybrid(m,cut=cut)
                ctl.bind_question_tokens(data['input_ids'],mask(data,'task_content'))
                out=m.generate(**data,**GEN);r=ctl.last_record
                n2=round(Fraction(32*8-cut*16,32-cut))
                self.assertEqual(r['layer_visual_counts'],[16]*cut+[n2]*(32-cut))
                self.assertEqual([e['after_layer'] for e in r['stage_events']],[0,cut])
                self.assertEqual(r['score_source_layer'],cut)
                self.assertEqual(r['drop_after_layer'],cut)
                source='full_attention_qk' if cut%4==0 else 'linear_postconv_qk'
                self.assertEqual(r['score_source_type'],source)
                c=r['native_hybrid_capture'];self.assertEqual(c,r['score_capture'])
                self.assertEqual(c['source_type'],source)
                self.assertEqual(c['source_layer'],cut)
                self.assertEqual(c['drop_after_layer'],cut)
                self.assertEqual(c['query_count'],4)
                self.assertEqual(c['score_query_original_positions'],list(range(data['input_ids'].shape[1]-4,data['input_ids'].shape[1])))
                self.assertEqual(r['effective_beta_mode'],'none' if cut%4==0 else 'native_beta')
                self.assertEqual(r['actual_prune_stages'],2)
                self.assertEqual(r['decode_steps'],2)
                for i in range(3,32,4):
                    self.assertEqual(out.past_key_values.get_seq_length(i),text_count+r['layer_visual_counts'][i]+2)
                for i in range(32):
                    if i%4!=3:
                        self.assertIsNotNone(out.past_key_values.conv_states[i])
                        self.assertIsNotNone(out.past_key_values.recurrent_states[i])
                self.assertIsNotNone(r['s2_pre_hidden'])
                ctl.remove()

    @torch.no_grad()
    def test_binding_keepall_consumption_and_cross_size_reset(self):
        for cut in (1,3,4,12,15):
            m=model();ctl=self.hybrid(m,target=8,cut=cut);seen=[]
            for data in (inputs(12),inputs(4),inputs(12)):
                ctl.bind_question_tokens(data['input_ids'],mask(data,'task_content'))
                out=m.generate(**data,**GEN)
                seen.append((out.sequences.clone(),copy.deepcopy(ctl.last_record)))
            torch.testing.assert_close(seen[0][0],seen[2][0],rtol=0,atol=0)
            self.assertEqual(seen[0][1],seen[2][1])
            self.assertEqual(seen[1][1]['actual_prune_stages'],0)
            self.assertIsNone(seen[1][1]['score_capture'])
            self.assertIn('score_query_binding',seen[1][1])
            with self.assertRaises(ValueError):m.generate(**inputs(12),**GEN)
            ctl.remove()

    def test_invalid_configuration_does_not_install_hooks(self):
        cases=[dict(cut_after_layer=16,quality_mode='native_hybrid_task_v1'),
               dict(cut_after_layer=6,quality_mode='qk'),
               dict(cut_after_layer=4,quality_mode='native_hybrid_task_v1',score_beta_mode='unit'),
               dict(cut_after_layer=4,quality_mode='native_hybrid_task_v1',score_query_scope='all_nonvisual')]
        for kw in cases:
            m=model();before=m.model.language_model.forward
            with self.assertRaises(ValueError):install_pruning(m,'mico',8,**kw)
            self.assertFalse(hasattr(m,'_qwen35_pruning_controller'))
            self.assertEqual(before,m.model.language_model.forward)

    @torch.no_grad()
    def test_early_cut_decode_retains_original_logical_positions(self):
        data=inputs(12)
        for cut in (1,2,3):
            m=model();positions,_=m.model.get_rope_index(data['input_ids'],
                image_grid_thw=data['image_grid_thw'],attention_mask=data['attention_mask'])
            observed={i:[] for i in range(32)};handles=[]
            for i,layer in enumerate(m.model.language_model.layers):
                def capture(module,args,kwargs,i=i):
                    if args[0].shape[1]==1:
                        observed[i].append(kwargs['position_ids'].detach().clone())
                handles.append(layer.register_forward_pre_hook(capture,with_kwargs=True))
            ctl=self.hybrid(m,cut=cut)
            ctl.bind_question_tokens(data['input_ids'],mask(data,'task_content'))
            m.generate(**data,**GEN)
            for i in range(32):
                self.assertEqual(len(observed[i]),2)
                for step,value in enumerate(observed[i],start=1):
                    torch.testing.assert_close(value,positions[...,-1:]+step,rtol=0,atol=0)
            for handle in handles:handle.remove()
            ctl.remove()

    @torch.no_grad()
    def test_legacy_qk_four_eight_exact(self):
        if not os.environ.get('QWEN35_LEGACY_CORE'):
            self.skipTest('Optional external historical-core parity test')
        source=Path(os.environ['QWEN35_LEGACY_CORE'])
        spec=importlib.util.spec_from_file_location('legacy_layer_core',source)
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        data=inputs(12)
        for cut in (4,8):
            old=model();new=copy.deepcopy(old)
            a=legacy.install_pruning(old,'mico',8,cut_after_layer=cut)
            b=install_pruning(new,'mico',8,cut_after_layer=cut)
            expected=old.generate(**data,**GEN);actual=new.generate(**data,**GEN)
            torch.testing.assert_close(actual.sequences,expected.sequences,rtol=0,atol=0)
            for x,y in zip(actual.logits,expected.logits):torch.testing.assert_close(x,y,rtol=0,atol=0)
            for name in ('key_cache','value_cache','conv_states','recurrent_states'):
                for x,y in zip(getattr(actual.past_key_values,name),getattr(expected.past_key_values,name)):
                    if x is None or y is None:self.assertIs(x,y)
                    else:torch.testing.assert_close(x,y,rtol=0,atol=0)
            self.assertEqual(a.last_record['stage_events'],b.last_record['stage_events'])
            a.remove();b.remove()

    @torch.no_grad()
    def test_qk_twelve_and_s1_have_distinct_trace(self):
        m=model();data=inputs(12)
        ctl=install_pruning(m,'mico',8,cut_after_layer=12)
        m.generate(**data,**GEN)
        self.assertEqual(ctl.last_record['score_source_layer'],12)
        self.assertEqual(ctl.last_record['score_query_scope'],'all_nonvisual')
        ctl.remove()
        ctl=install_pruning(m,'mico_s1',8)
        m.generate(**data,**GEN)
        self.assertIsNone(ctl.last_record['score_source_layer'])
        self.assertIsNone(ctl.last_record['drop_after_layer'])
        self.assertEqual(ctl.last_record['layer_visual_counts'],[8]*32)
        ctl.remove()


if __name__=='__main__':unittest.main()
