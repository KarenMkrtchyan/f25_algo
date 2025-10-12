# Interpretability Functions Test Suite

This directory contains comprehensive test cases for the interpretability functions in the `src/Interpretability/` module.

## Test Files

### `test_simple.py`
Basic test cases for functions that can be tested without complex dependencies:
- `hook_function` - Tests that hook functions return input unchanged
- `get_induction_score_store` - Tests tensor creation for induction scores
- `head_zero_ablation_hook` - Tests zero ablation of attention heads
- `head_mean_ablation_hook` - Tests mean ablation of attention heads
- `logit_attribution` - Tests logit attribution calculations
- `attention_head_display` - Tests attention head visualization
- `attention_pattern_display` - Tests attention pattern visualization

### `test_interpretability.py`
Comprehensive test cases with mocking for complex functions:
- Integration tests with model loading
- Edge case testing
- Error condition testing
- Mock-based testing for functions requiring external dependencies

## Running Tests

### Run All Tests
```bash
cd /path/to/f25_algo
source venv/bin/activate
PYTHONPATH=src python src/run_tests.py
```

### Run Specific Test File
```bash
cd /path/to/f25_algo
source venv/bin/activate
PYTHONPATH=src python -m unittest src.tests.test_simple
```

### Run Individual Test Class
```bash
cd /path/to/f25_algo
source venv/bin/activate
PYTHONPATH=src python -m unittest src.tests.test_simple.TestBasicFunctions
```

## Test Coverage

The tests cover:

✅ **Basic Functions**
- Hook function behavior
- Tensor creation and manipulation
- Ablation hook functionality
- Logit attribution calculations

✅ **Edge Cases**
- Zero dimensions
- Large dimensions
- Single head scenarios
- Minimal tensor sizes

✅ **Integration**
- Model loading compatibility
- Tensor operation consistency
- Function interaction testing

✅ **Error Handling**
- Invalid input detection
- Proper error messages
- Graceful failure handling

## Test Results

When all tests pass, you should see:
```
✅ All tests passed!
🎉 All tests completed successfully!
✅ Your interpretability functions are working correctly.
📝 You can now use these functions in your experiments.
```

## Examples

See `src/examples/interpretability_example.py` for practical usage examples of the interpretability functions.

## Notes

- Some functions in the original code have undefined variables (like `seq_len`, `induction_score_store` in `induction_score_hook`)
- These issues are noted in the tests and would need to be fixed in the original functions
- The tests focus on functions that can be properly tested with the current implementation
- Mock objects are used to simulate complex dependencies like model caches and visualization libraries
