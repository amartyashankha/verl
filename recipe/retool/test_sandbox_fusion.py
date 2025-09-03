# ---
# cmd: ["modal", "run", "recipe/retool/test_sandbox_fusion.py"]
# ---

# # Test Sandbox Fusion Service

# This script tests the deployed Sandbox Fusion service from a separate Modal app.

import json
from typing import Any, Dict

import httpx
import modal

app = modal.App("test-sandbox-fusion")

# Create image with httpx dependency
image = modal.Image.debian_slim().pip_install("httpx")

# Update this URL after deploying modal_fusion_sandbox.py
SANDBOX_URL = "https://shortcut--sandbox-fusion.modal.run"


def print_result(test_name: str, result: Dict[str, Any], success: bool):
    """Pretty print test results."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"\n{status} {test_name}")
    print(f"Response: {json.dumps(result, indent=2)}")


@app.function(image=image)
def run_tests():
    """Run comprehensive tests on the Sandbox Fusion service."""

    print(f"🧪 Testing Sandbox Fusion Service at: {SANDBOX_URL}")

    if "YOUR-WORKSPACE" in SANDBOX_URL:
        print("\n⚠️  ERROR: Please update SANDBOX_URL with your actual deployment URL!")
        print(
            "After deploying modal_fusion_sandbox.py, update the SANDBOX_URL constant."
        )
        return

    print("\n1️⃣ Testing simple Python execution...")
    try:
        response = httpx.post(
            f"{SANDBOX_URL}",
            json={"code": "print('Hello from sandbox!')", "language": "python"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        success = result.get(
            "status"
        ) == "Success" and "Hello from sandbox!" in result.get("run_result", {}).get(
            "stdout", ""
        )
        print_result("Simple Python", result, success)
    except Exception as e:
        print_result("Simple Python", {"error": str(e)}, False)

    # Test 3: Math computation
    print("\n3️⃣ Testing math computation...")
    try:
        code = """
import math
result = math.sqrt(16) + math.pi
print(f"Result: {result:.4f}")
"""
        response = httpx.post(
            f"{SANDBOX_URL}",
            json={"code": code, "language": "python"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        success = result.get("status") == "Success" and "Result: 7.1416" in result.get(
            "run_result", {}
        ).get("stdout", "")
        print_result("Math Computation", result, success)
    except Exception as e:
        print_result("Math Computation", {"error": str(e)}, False)

    # Test 4: Multi-line code with imports
    print("\n4️⃣ Testing multi-line code with imports...")
    try:
        code = """
import json
import datetime

data = {
    "timestamp": datetime.datetime.now().isoformat(),
    "message": "Testing from Modal",
    "numbers": [1, 2, 3, 4, 5]
}

print(json.dumps(data, indent=2))
print(f"Sum of numbers: {sum(data['numbers'])}")
"""
        response = httpx.post(
            f"{SANDBOX_URL}",
            json={"code": code, "language": "python"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        success = result.get(
            "status"
        ) == "Success" and "Sum of numbers: 15" in result.get("run_result", {}).get(
            "stdout", ""
        )
        print_result("Multi-line with Imports", result, success)
    except Exception as e:
        print_result("Multi-line with Imports", {"error": str(e)}, False)

    # Test 5: Error handling
    print("\n5️⃣ Testing error handling...")
    try:
        code = """
# This should produce an error
x = 1 / 0
"""
        response = httpx.post(
            f"{SANDBOX_URL}",
            json={"code": code, "language": "python"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        success = (
            result.get("status") == "Success"  # API call succeeds
            and result.get("run_result", {}).get("return_code", 0)
            != 0  # But code fails
            and "ZeroDivisionError" in result.get("run_result", {}).get("stderr", "")
        )
        print_result("Error Handling", result, success)
    except Exception as e:
        print_result("Error Handling", {"error": str(e)}, False)

    # Test 6: Timeout behavior (optional - takes time)
    print("\n6️⃣ Testing timeout behavior (this may take a while)...")
    try:
        code = """
import time
print("Starting long computation...")
time.sleep(35)  # Longer than default timeout
print("This should not print")
"""
        response = httpx.post(
            f"{SANDBOX_URL}",
            json={"code": code, "language": "python", "timeout": 5.0},
            timeout=10,  # Client timeout shorter than sleep
        )
        # We expect this to timeout
        print_result("Timeout Test", {"note": "Timeout expected"}, False)
    except httpx.TimeoutException:
        print_result("Timeout Test", {"result": "Correctly timed out"}, True)
    except Exception as e:
        print_result("Timeout Test", {"error": str(e)}, False)

    print("\n🏁 Test suite completed!")


@app.function(image=image)
def test_single_request(code: str = "print('Quick test!')"):
    """Test a single code execution request."""

    if "YOUR-WORKSPACE" in SANDBOX_URL:
        print("⚠️  ERROR: Please update SANDBOX_URL with your actual deployment URL!")
        return

    print(f"🧪 Testing code execution: {code}")

    try:
        response = httpx.post(
            f"{SANDBOX_URL}",
            json={"code": code, "language": "python"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        print("\n📊 Response:")
        print(json.dumps(result, indent=2))

        if result.get("status") == "Success":
            print(f"\n✅ Success! Output: {result['run_result']['stdout'].strip()}")
        else:
            print(f"\n❌ Failed: {result.get('message', 'Unknown error')}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


@app.local_entrypoint()
def main():
    """Run the test suite."""
    print("🚀 Sandbox Fusion Test Suite\n")

    print("Usage:")
    print("  # Run full test suite")
    print("  modal run recipe/retool/test_sandbox_fusion.py")
    print("")
    print("  # Test single code snippet")
    print(
        "  modal run recipe/retool/test_sandbox_fusion.py::test_single_request --code 'print(2+2)'"
    )
    print("")
    print("Before running tests:")
    print("1. Deploy the sandbox service:")
    print("   modal deploy recipe/retool/modal_fusion_sandbox.py")
    print("")
    print("2. Update SANDBOX_URL in this file with your deployment URL")
    print("")
    print("Running test suite...\n")

    run_tests.remote()


# Additional test scenarios to consider:
# 1. Test with different languages (if supported)
# 2. Test file operations and sandboxing
# 3. Test resource limits (memory, CPU)
# 4. Test concurrent requests
# 5. Test with malicious code (security)
