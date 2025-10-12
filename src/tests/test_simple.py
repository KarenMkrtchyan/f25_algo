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
    logit_attribution
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

    def test_hook_function(self):
        """Test hook_function returns input unchanged."""
        result = hook_function(self, self.attn_pattern, self.mock_hook)
        
        # Should return the input pattern unchanged
        t.testing.assert_close(result, self.attn_pattern)
        self.assertEqual(result.shape, self.attn_pattern.shape)
        self.assertIsInstance(result, t.Tensor)

    def test_get_induction_score_store(self):
        """Test get_induction_score_store creates correct tensor."""
        # Test basic functionality
        result = get_induction_score_store(self.n_layers, self.n_heads, self.device)
        
        self.assertEqual(result.shape, (self.n_layers, self.n_heads))
        self.assertEqual(result.device.type, self.device)
        self.assertTrue(t.all(result == 0))
        self.assertIsInstance(result, t.Tensor)
        
        # Test with different parameters
        result_2 = get_induction_score_store(4, 12, self.device)
        self.assertEqual(result_2.shape, (4, 12))
        self.assertTrue(t.all(result_2 == 0))

    def test_head_zero_ablation_hook(self):
        """Test head_zero_ablation_hook zeros out specified head."""
        # Create a copy to avoid modifying the original
        z_copy = self.z_tensor.clone()
        head_to_ablate = 3
        
        # Apply the hook
        head_zero_ablation_hook(z_copy, self.mock_hook, head_to_ablate)
        
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
        head_mean_ablation_hook(z_copy, self.mock_hook, head_to_ablate)
        
        # Check that the specified head is replaced with mean
        self.assertTrue(t.allclose(z_copy[:, :, head_to_ablate, :], expected_mean))

    def test_logit_attribution(self):
        """Test logit_attribution function."""
        result = logit_attribution(
            self.embed,
            self.l1_results,
            self.l2_results,
            self.W_U,
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
        W_U = t.randn(10, 5)
        tokens = t.tensor([0, 1])
        
        result = logit_attribution(embed, l1_results, l2_results, W_U, tokens)
        self.assertEqual(result.shape, (1, 3))  # seq-1, 1+2*1
        self.assertTrue(t.all(t.isfinite(result)))

    def test_ablation_hooks_edge_cases(self):
        """Test ablation hooks with edge cases."""
        # Test with single head
        z_single_head = t.randn(1, 5, 1, 10)
        mock_hook = Mock()
        
        # Zero ablation
        z_copy = z_single_head.clone()
        head_zero_ablation_hook(z_copy, mock_hook, 0)
        self.assertTrue(t.all(z_copy[:, :, 0, :] == 0))
        
        # Mean ablation
        z_copy = z_single_head.clone()
        expected_mean = z_single_head[:, :, 0, :].mean(0)
        head_mean_ablation_hook(z_copy, mock_hook, 0)
        self.assertTrue(t.allclose(z_copy[:, :, 0, :], expected_mean))

    def test_get_induction_score_store_edge_cases(self):
        """Test get_induction_score_store with edge cases."""
        # Test with zero dimensions
        result = get_induction_score_store(0, 0, self.device)
        self.assertEqual(result.shape, (0, 0))
        
        # Test with large dimensions
        result = get_induction_score_store(100, 100, self.device)
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
        result_hook = hook_function(self, attn_pattern, mock_hook)
        self.assertEqual(result_hook.shape, attn_pattern.shape)
        
        # Test ablation hooks
        z_copy = z_tensor.clone()
        head_zero_ablation_hook(z_copy, mock_hook, 0)
        self.assertTrue(t.all(z_copy[:, :, 0, :] == 0))
        
        # Test mean ablation
        z_copy = z_tensor.clone()
        original_head = z_tensor[:, :, 0, :].clone()
        head_mean_ablation_hook(z_copy, mock_hook, 0)
        expected_mean = original_head.mean(0)
        self.assertTrue(t.allclose(z_copy[:, :, 0, :], expected_mean))


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
