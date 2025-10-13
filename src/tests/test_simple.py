"""
Simple test cases for interpretability functions.
This module contains basic tests for functions that can be tested without complex dependencies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import Mock
import torch as t
import numpy as np

# Import the functions to test
from Interpretability.Functions import (
    hook_function,
    get_induction_score_store,
    head_zero_ablation_hook,
    head_mean_ablation_hook,
    logit_attribution,
    visualize_neuron_activation,
    display_attention_heads,
    display_attention_patterns
)

from Interpretability.Attention_display import attention_head_display, attention_pattern_display


class TestBasicFunctions(unittest.TestCase):
    """Test cases for basic interpretability functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.batch_size = 2
        self.seq_len = 10
        self.n_heads = 8
        self.d_head = 64
        self.d_model = 512
        self.n_layers = 6
        self.d_vocab = 1000
        
        # Create test tensors
        self.attn_pattern = t.randn(self.batch_size, self.n_heads, self.seq_len, self.seq_len)
        self.z_tensor = t.randn(self.batch_size, self.seq_len, self.n_heads, self.d_head)
        self.embed = t.randn(self.seq_len, self.d_model)
        self.l1_results = t.randn(self.seq_len, self.n_heads, self.d_model)
        self.l2_results = t.randn(self.seq_len, self.n_heads, self.d_model)
        self.W_U = t.randn(self.d_model, self.d_vocab)
        self.tokens = t.randint(0, self.d_vocab, (self.seq_len,))
        
        # Create mock hook
        self.mock_hook = Mock()
        self.mock_hook.layer.return_value = 2
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.cfg.n_layers = self.n_layers
        self.mock_model.cfg.n_heads = self.n_heads
        self.mock_model.cfg.device = self.device
        self.mock_model.W_U = t.randn(self.d_model, self.d_vocab)

    def test_hook_function(self):
        """Test hook_function returns input unchanged."""
        result = hook_function(self.mock_model, self.attn_pattern, self.mock_hook)
        
        # Should return the input pattern unchanged
        t.testing.assert_close(result, self.attn_pattern)
        self.assertEqual(result.shape, self.attn_pattern.shape)
        self.assertIsInstance(result, t.Tensor)

    def test_get_induction_score_store(self):
        """Test get_induction_score_store creates correct tensor."""
        # Test basic functionality
        result = get_induction_score_store(self.mock_model)
        
        self.assertEqual(result.shape, (self.mock_model.cfg.n_layers, self.mock_model.cfg.n_heads))
        self.assertEqual(result.device.type, self.device)
        self.assertTrue(t.all(result == 0))
        self.assertIsInstance(result, t.Tensor)

    def test_head_zero_ablation_hook(self):
        """Test head_zero_ablation_hook zeros out specified head."""
        # Create a copy to avoid modifying the original
        z_copy = self.z_tensor.clone()
        head_to_ablate = 3
        
        # Apply the hook
        head_zero_ablation_hook(self.mock_model, z_copy, self.mock_hook, head_to_ablate)
        
        # Check that the specified head is zeroed
        self.assertTrue(t.all(z_copy[:, :, head_to_ablate, :] == 0))
        
        # Check that other heads are unchanged
        for head in range(self.n_heads):
            if head != head_to_ablate:
                self.assertTrue(t.allclose(z_copy[:, :, head, :], self.z_tensor[:, :, head, :]))

    def test_head_mean_ablation_hook(self):
        """Test head_mean_ablation_hook replaces head with mean."""
        # Create a copy to avoid modifying the original
        z_copy = self.z_tensor.clone()
        head_to_ablate = 3
        
        # Calculate expected mean
        expected_mean = self.z_tensor[:, :, head_to_ablate, :].mean(0)
        
        # Apply the hook
        head_mean_ablation_hook(self.mock_model, z_copy, self.mock_hook, head_to_ablate)
        
        # Check that the specified head is replaced with mean
        self.assertTrue(t.allclose(z_copy[:, :, head_to_ablate, :], expected_mean))

    def test_logit_attribution(self):
        """Test logit_attribution function."""
        result = logit_attribution(
            self.mock_model,
            self.embed,
            self.l1_results,
            self.l2_results,
            self.tokens
        )
        
        # Check output shape
        expected_seq_len = self.seq_len - 1
        expected_n_components = 1 + 2 * self.n_heads
        self.assertEqual(result.shape, (expected_seq_len, expected_n_components))
        self.assertIsInstance(result, t.Tensor)
        
        # Check that result has reasonable values (not NaN or Inf)
        self.assertTrue(t.all(t.isfinite(result)))

    def test_logit_attribution_edge_cases(self):
        """Test logit_attribution with edge cases."""
        # Test with minimal dimensions
        embed = t.randn(2, 10)
        l1_results = t.randn(2, 1, 10)
        l2_results = t.randn(2, 1, 10)
        tokens = t.tensor([0, 1])
        
        # Create a mock model with the right W_U shape
        mock_model = Mock()
        mock_model.W_U = t.randn(10, 5)
        
        result = logit_attribution(mock_model, embed, l1_results, l2_results, tokens)
        self.assertEqual(result.shape, (1, 3))  # seq-1, 1+2*1
        self.assertTrue(t.all(t.isfinite(result)))

    def test_ablation_hooks_edge_cases(self):
        """Test ablation hooks with edge cases."""
        # Test with single head
        z_single_head = t.randn(1, 5, 1, 10)
        mock_hook = Mock()
        mock_model = Mock()
        
        # Zero ablation
        z_copy = z_single_head.clone()
        head_zero_ablation_hook(mock_model, z_copy, mock_hook, 0)
        self.assertTrue(t.all(z_copy[:, :, 0, :] == 0))
        
        # Mean ablation
        z_copy = z_single_head.clone()
        expected_mean = z_single_head[:, :, 0, :].mean(0)
        head_mean_ablation_hook(mock_model, z_copy, mock_hook, 0)
        self.assertTrue(t.allclose(z_copy[:, :, 0, :], expected_mean))

    def test_get_induction_score_store_edge_cases(self):
        """Test get_induction_score_store with edge cases."""
        # Test with different model configurations
        mock_model_small = Mock()
        mock_model_small.cfg.n_layers = 0
        mock_model_small.cfg.n_heads = 0
        mock_model_small.cfg.device = self.device
        
        result = get_induction_score_store(mock_model_small)
        self.assertEqual(result.shape, (0, 0))
        
        # Test with large dimensions
        mock_model_large = Mock()
        mock_model_large.cfg.n_layers = 100
        mock_model_large.cfg.n_heads = 100
        mock_model_large.cfg.device = self.device
        
        result = get_induction_score_store(mock_model_large)
        self.assertEqual(result.shape, (100, 100))
        self.assertTrue(t.all(result == 0))


class TestAttentionDisplay(unittest.TestCase):
    """Test cases for attention display functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_heads = 8
        self.seq_len = 10
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.to_str_tokens.return_value = ["hello", "world", "test", "example"] * 3
        
        # Create mock cache with proper structure
        class MockCache:
            def __init__(self, n_heads, seq_len):
                self.n_heads = n_heads
                self.seq_len = seq_len
                
            def __getitem__(self, key):
                if key == ("pattern", 0):
                    return t.randn(1, self.n_heads, self.seq_len, self.seq_len)
                raise KeyError(key)
        
        self.mock_cache = MockCache(self.n_heads, self.seq_len)

    def test_attention_head_display_basic(self):
        """Test attention_head_display basic functionality."""
        # Mock circuitsvis
        with unittest.mock.patch('Interpretability.Attention_display.cv') as mock_cv:
            mock_cv.attention.attention_heads.return_value = "attention_vis"
            
            result = attention_head_display(
                self.mock_model, 
                "hello world", 
                self.mock_cache, 
                layer=0, 
                position=0
            )
            
            self.assertEqual(result, "attention_vis")
            self.mock_model.to_str_tokens.assert_called_once_with("hello world")
            mock_cv.attention.attention_heads.assert_called_once()

    def test_attention_pattern_display_basic(self):
        """Test attention_pattern_display basic functionality."""
        # Mock circuitsvis
        with unittest.mock.patch('Interpretability.Attention_display.cv') as mock_cv:
            mock_cv.attention.attention_patterns.return_value = "pattern_vis"
            
            result = attention_pattern_display(
                self.mock_model, 
                "hello world", 
                self.mock_cache, 
                layer=0, 
                position=0, 
                num_heads=self.n_heads
            )
            
            self.assertEqual(result, "pattern_vis")
            self.mock_model.to_str_tokens.assert_called_once_with("hello world")
            mock_cv.attention.attention_patterns.assert_called_once()


class TestTensorOperations(unittest.TestCase):
    """Test tensor operations and consistency."""
    
    def test_tensor_consistency(self):
        """Test that tensor operations are consistent."""
        batch_size, seq_len, n_heads, d_head = 2, 10, 8, 64
        
        # Create consistent tensors
        attn_pattern = t.randn(batch_size, n_heads, seq_len, seq_len)
        z_tensor = t.randn(batch_size, seq_len, n_heads, d_head)
        
        mock_hook = Mock()
        
        # Test hook function
        mock_model = Mock()
        result_hook = hook_function(mock_model, attn_pattern, mock_hook)
        self.assertEqual(result_hook.shape, attn_pattern.shape)
        
        # Test ablation hooks
        z_copy = z_tensor.clone()
        head_zero_ablation_hook(mock_model, z_copy, mock_hook, 0)
        self.assertTrue(t.all(z_copy[:, :, 0, :] == 0))
        
        # Test mean ablation
        z_copy = z_tensor.clone()
        original_head = z_tensor[:, :, 0, :].clone()
        head_mean_ablation_hook(mock_model, z_copy, mock_hook, 0)
        expected_mean = original_head.mean(0)
        self.assertTrue(t.allclose(z_copy[:, :, 0, :], expected_mean))

class TestNeuronAndAttentionVisualization(unittest.TestCase):
    """Tests for new neuron and attention visualization functions."""

    def setUp(self):
        """Set up a mock model, cache, and tokens."""
        self.n_layers = 4
        self.n_heads = 8
        self.seq_len = 10
        self.n_neurons = 16

        # Mock HookedTransformer model
        self.mock_model = Mock()
        self.mock_model.cfg.n_layers = self.n_layers
        self.mock_model.cfg.n_heads = self.n_heads

        # Mock to_str_tokens
        self.mock_model.to_str_tokens.side_effect = lambda x: ["token" + str(i) for i in range(self.seq_len)]

        # Mock cache with "post" activations and "pattern" for attention
        class MockCache:
            def __getitem__(self_inner, key):
                if key[0] == "post":
                    return t.randn(self.seq_len, self.n_neurons)
                elif key[0] == "pattern":
                    return t.randn(1, self.n_heads, self.seq_len, self.seq_len)
                raise KeyError(key)

        self.mock_cache = MockCache()
        self.str_tokens = ["token" + str(i) for i in range(self.seq_len)]

    @unittest.mock.patch('Interpretability.Functions.cv')
    def test_visualize_neuron_activation(self, mock_cv):
        """Test neuron activation visualization."""
        mock_cv.activations.text_neuron_activations.return_value = "text_vis"
        mock_cv.topk_tokens.topk_tokens.return_value = "topk_vis"

        text_vis, topk_vis, activations, activations_rearranged = visualize_neuron_activation(
            self.mock_cache, self.mock_model, self.str_tokens, max_k=5
        )

        mock_cv.activations.text_neuron_activations.assert_called_once()
        mock_cv.topk_tokens.topk_tokens.assert_called_once()

        self.assertEqual(text_vis, "text_vis")
        self.assertEqual(topk_vis, "topk_vis")
        self.assertEqual(activations.shape, (self.seq_len, self.n_layers, self.n_neurons))
        self.assertEqual(activations_rearranged.shape, (1, self.n_layers, self.seq_len, self.n_neurons))

    @unittest.mock.patch('Interpretability.Functions.cv')
    def test_display_attention_heads(self, mock_cv):
        """Test attention heads visualization."""
        mock_cv.attention.attention_heads.return_value = "attention_heads_vis"

        vis = display_attention_heads(self.mock_model, "dummy text", self.mock_cache, layer=0, position=2)

        self.mock_model.to_str_tokens.assert_called_with("dummy text")
        mock_cv.attention.attention_heads.assert_called_once()
        self.assertEqual(vis, "attention_heads_vis")

    @unittest.mock.patch('Interpretability.Functions.cv')
    def test_display_attention_patterns(self, mock_cv):
        """Test attention patterns visualization."""
        mock_cv.attention.attention_patterns.return_value = "attention_patterns_vis"

        vis = display_attention_patterns(self.mock_model, "dummy text", self.mock_cache, layer=1, position=0)

        self.mock_model.to_str_tokens.assert_called_with("dummy text")
        mock_cv.attention.attention_patterns.assert_called_once()
        self.assertEqual(vis, "attention_patterns_vis")

    @unittest.mock.patch('Interpretability.Functions.cv')
    def test_attention_heads_with_tokens_list(self, mock_cv):
        """Test attention heads with pre-tokenized input list."""
        mock_cv.attention.attention_heads.return_value = "attention_heads_vis"

        tokens_list = ["token" + str(i) for i in range(self.seq_len)]
        vis = display_attention_heads(self.mock_model, tokens_list, self.mock_cache, layer=0, position=1)

        self.mock_model.to_str_tokens.assert_called_with(tokens_list)
        self.assertEqual(vis, "attention_heads_vis")

    @unittest.mock.patch('Interpretability.Functions.cv')
    def test_attention_patterns_with_tokens_list(self, mock_cv):
        """Test attention patterns with pre-tokenized input list."""
        mock_cv.attention.attention_patterns.return_value = "attention_patterns_vis"

        tokens_list = ["token" + str(i) for i in range(self.seq_len)]

def run_simple_tests():
    """Run simple test cases."""
    print("Running simple interpretability function tests...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestBasicFunctions))
    test_suite.addTest(unittest.makeSuite(TestAttentionDisplay))
    test_suite.addTest(unittest.makeSuite(TestTensorOperations))
    test_suite.addTest(unittest.makeSuite(TestNeuronAndAttentionVisualization))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures + result.errors)} test(s) failed.")
        return False

if __name__ == "__main__":
    success = run_simple_tests()
    exit(0 if success else 1)
