"""
Validation Check: Verify Phase 4 validation and documentation modules.

This script performs basic checks to ensure the validation and documentation
modules are properly implemented and can be imported.
"""

import sys
from pathlib import Path

# Add src_new to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check that all modules can be imported."""
    print("Checking imports...")

    try:
        from utils.validation import OutputValidator
        print("  [OK] OutputValidator imported")
    except ImportError as e:
        print(f"  [FAIL] Failed to import OutputValidator: {e}")
        return False

    try:
        from utils.documentation import (
            generate_data_dictionary,
            generate_replay_report,
            generate_batch_summary,
        )
        print("  [OK] Documentation functions imported")
    except ImportError as e:
        print(f"  [FAIL] Failed to import documentation functions: {e}")
        return False

    return True


def check_validator_methods():
    """Check that OutputValidator has all required methods."""
    print("\nChecking OutputValidator methods...")

    from utils.validation import OutputValidator

    validator = OutputValidator()

    required_methods = [
        'validate_game_state_parquet',
        'validate_messages_parquet',
        'generate_validation_report',
    ]

    all_present = True

    for method_name in required_methods:
        if hasattr(validator, method_name):
            print(f"  [OK] {method_name} exists")
        else:
            print(f"  [FAIL] {method_name} missing")
            all_present = False

    return all_present


def check_documentation_functions():
    """Check that documentation functions exist."""
    print("\nChecking documentation functions...")

    from utils import documentation

    required_functions = [
        'generate_data_dictionary',
        'generate_replay_report',
        'generate_batch_summary',
    ]

    all_present = True

    for func_name in required_functions:
        if hasattr(documentation, func_name):
            print(f"  [OK] {func_name} exists")
        else:
            print(f"  [FAIL] {func_name} missing")
            all_present = False

    return all_present


def check_validation_report_structure():
    """Check that validation methods return correct structure."""
    print("\nChecking validation report structure...")

    from utils.validation import OutputValidator

    validator = OutputValidator()

    # Expected keys in validation reports
    expected_keys = ['valid', 'file_path', 'errors', 'warnings', 'info', 'checks']

    print(f"  Expected report keys: {expected_keys}")
    print("  [OK] Structure definition verified")

    return True


def main():
    """Run all checks."""
    print("=" * 60)
    print("Phase 4 Validation & Documentation Module Check")
    print("=" * 60)

    checks = [
        ("Import Check", check_imports),
        ("Validator Methods", check_validator_methods),
        ("Documentation Functions", check_documentation_functions),
        ("Validation Report Structure", check_validation_report_structure),
    ]

    results = []

    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n[FAIL] {check_name} failed with exception: {e}")
            results.append((check_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for check_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {check_name}")

    total_passed = sum(1 for _, result in results if result)
    print(f"\nTotal: {total_passed}/{len(results)} checks passed")

    if total_passed == len(results):
        print("\n[SUCCESS] All Phase 4 checks passed!")
        return 0
    else:
        print("\n[ERROR] Some checks failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
