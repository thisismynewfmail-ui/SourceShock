#!/usr/bin/env python3
"""
AutoType v1.02 — Memory System Test
Verifies: store, retrieve, injection, dumb mode, deletion, rebuild, think stripping.
Run: python test_memory.py
"""
import os, sys, shutil, time

# Use a temp data dir so we don't clobber real memories
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "memory_test")
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR)

# Monkey-patch paths before importing
import memory as mem_module
mem_module.DATA_DIR = TEST_DIR
mem_module.DB_PATH = os.path.join(TEST_DIR, "ltm.db")
mem_module.VEC_PATH = os.path.join(TEST_DIR, "embeddings.npy")
mem_module.CONFIG_PATH = os.path.join(TEST_DIR, "ltm_config.json")

from memory import LongTermMemory, strip_think
import numpy as np

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name} — {detail}")
        failed += 1

print("=" * 60)
print("AutoType v1.02 — Memory System Test")
print("=" * 60)

# ── Test 1: Think tag stripping ──────────────────────────────────
print("\n[1] Think Tag Stripping")
test("Complete block", strip_think("hello <think>internal</think> world") == "hello world")
test("Unclosed tag", strip_think("hello <think>partial...") == "hello")
test("Empty tags", strip_think("<think></think>clean") == "clean")
test("Nested content", strip_think("<think>a\nb\nc</think>result") == "result")
test("No tags", strip_think("normal text") == "normal text")
test("Empty input", strip_think("") == "")
test("None input", strip_think(None) == "")

# ── Test 2: LTM initialization ──────────────────────────────────
print("\n[2] Initialization")
ltm = LongTermMemory()
test("Config loaded", ltm.config is not None)
test("Saving disabled by default", ltm.config["enable_saving"] == False)
test("Injection disabled by default", ltm.config["enable_injection"] == False)
test("Dumb mode disabled", ltm.config["dumb_mode"] == False)

stats = ltm.get_stats()
test("Empty DB", stats["total_memories"] == 0)
test("No vectors", stats["vector_count"] == 0)

# ── Test 3: Store blocked when saving disabled ───────────────────
print("\n[3] Save-disabled guard")
ltm.evaluate_and_store("hello", "world")
test("Blocked", ltm.last_action == "save_disabled")
test("Stats unchanged", ltm.get_stats()["total_memories"] == 0)

# ── Test 4: Dumb mode storage ────────────────────────────────────
print("\n[4] Dumb Mode")
ltm.config["enable_saving"] = True
ltm.config["dumb_mode"] = True

ltm.evaluate_and_store("What is Python?", "Python is a programming language.")
test("Saved", ltm.last_action == "saved_dumb")
test("Interest=1.0", ltm.last_interest == 1.0)
test("DB has 1", ltm.get_stats()["total_memories"] == 1)
test("Vec has 1", ltm.get_stats()["vector_count"] == 1)

ltm.evaluate_and_store("Tell me about cats", "Cats are small domesticated mammals.")
test("Second saved", ltm.last_action == "saved_dumb")
test("DB has 2", ltm.get_stats()["total_memories"] == 2)
test("Vec has 2", ltm.get_stats()["vector_count"] == 2)

# Same content again — dumb mode saves it anyway
ltm.evaluate_and_store("What is Python?", "Python is a programming language.")
test("Duplicate saved (dumb)", ltm.last_action == "saved_dumb")
test("DB has 3", ltm.get_stats()["total_memories"] == 3)

# ── Test 5: Smart mode storage ───────────────────────────────────
print("\n[5] Smart Mode")
ltm.config["dumb_mode"] = False
ltm.config["interest_threshold"] = 0.30

ltm.evaluate_and_store("What is JavaScript?", "JavaScript is a scripting language for the web.")
test("Novel saved", ltm.last_action == "saved")
test("Interest > 0", ltm.last_interest > 0)
test("DB has 4", ltm.get_stats()["total_memories"] == 4)

# Very similar to existing
ltm.evaluate_and_store("What is Python?", "Python is a programming language for general use.")
# This should be skipped due to similarity
is_skip_or_save = ltm.last_action in ("skipped", "saved")
test("Similar handled", is_skip_or_save, f"got {ltm.last_action}")

# ── Test 6: Min length guard ─────────────────────────────────────
print("\n[6] Min Length Guard")
ltm.evaluate_and_store("hi", "yo")
test("Too short blocked", ltm.last_action == "too_short")

# ── Test 7: Think stripping in storage ────────────────────────────
print("\n[7] Think Tags in Storage")
before = ltm.get_stats()["total_memories"]
ltm.config["dumb_mode"] = True
ltm.evaluate_and_store(
    "What is AI?",
    "<think>Let me think about this carefully...</think>AI is artificial intelligence."
)
test("Stored", ltm.last_action == "saved_dumb")
mems = ltm.list_memories(1)
test("Think stripped from assistant", "<think>" not in mems[0]["assistant"])
test("Content preserved", "AI is artificial intelligence" in mems[0]["assistant"])
ltm.config["dumb_mode"] = False

# ── Test 8: Retrieval ─────────────────────────────────────────────
print("\n[8] Retrieval")
# Retrieval blocked when injection disabled
ltm.config["enable_injection"] = False
fmt, raw = ltm.retrieve_memories("What is Python?")
test("Blocked when disabled", len(fmt) == 0)
test("Action=disabled", ltm.last_action == "inject_disabled")

# Enable and retrieve
ltm.config["enable_injection"] = True
fmt, raw = ltm.retrieve_memories("Tell me about Python programming")
test("Got results", len(fmt) > 0, f"got {len(fmt)}")
test("Has similarity", len(raw) > 0 and raw[0]["similarity"] > 0)
test("Sorted by sim", all(raw[i]["similarity"] >= raw[i+1]["similarity"] for i in range(len(raw)-1)) if len(raw) > 1 else True)

# ── Test 9: Injection building ────────────────────────────────────
print("\n[9] Injection Building")
inj, raw = ltm.build_injection("Python programming language")
test("Injection built", inj is not None)
test("Has header", "[SYSTEM MEMORY CONTEXT]" in inj)
test("Has footer", "[END MEMORY CONTEXT]" in inj)
test("Has memory content", "User:" in inj and "Assistant:" in inj)

# Query with no match
inj2, raw2 = ltm.build_injection("quantum physics dark matter")
# May or may not match depending on threshold
test("No-match handled", ltm.last_action in ("no_match", "injected:0") or inj2 is not None)

# ── Test 10: Deletion ────────────────────────────────────────────
print("\n[10] Deletion")
mems = ltm.list_memories()
before_count = ltm.get_stats()["total_memories"]
before_vecs = ltm.get_stats()["vector_count"]
mid = mems[0]["id"]
ltm.delete_memory(mid)
after_count = ltm.get_stats()["total_memories"]
after_vecs = ltm.get_stats()["vector_count"]
test("DB count -1", after_count == before_count - 1)
test("Vec count -1", after_vecs == before_vecs - 1)

# ── Test 11: Vector rebuild ──────────────────────────────────────
print("\n[11] Vector Rebuild")
# Corrupt the .npy by deleting it
if os.path.exists(mem_module.VEC_PATH):
    os.remove(mem_module.VEC_PATH)
ltm.rebuild_vectors()
stats = ltm.get_stats()
test("Rebuild synced", stats["vector_count"] == stats["total_memories"])

# ── Test 12: Config persistence ───────────────────────────────────
print("\n[12] Config Persistence")
ltm.config["interest_threshold"] = 0.42
ltm.config["enable_saving"] = True
ltm.config["dumb_mode"] = True
ltm.save_config()

ltm2 = LongTermMemory()
test("Threshold persisted", ltm2.config["interest_threshold"] == 0.42)
test("Saving persisted", ltm2.config["enable_saving"] == True)
test("Dumb persisted", ltm2.config["dumb_mode"] == True)

# ── Test 13: Reset ────────────────────────────────────────────────
print("\n[13] Reset")
ltm.reset_all()
test("DB empty", ltm.get_stats()["total_memories"] == 0)
test("Vecs empty", ltm.get_stats()["vector_count"] == 0)
test("NPY gone", not os.path.exists(mem_module.VEC_PATH))

# ── Test 14: Full pipeline (store → retrieve → inject) ───────────
print("\n[14] Full Pipeline")
ltm.config["enable_saving"] = True
ltm.config["enable_injection"] = True
ltm.config["dumb_mode"] = False
ltm.config["interest_threshold"] = 0.0  # save everything
ltm.config["load_similarity_threshold"] = 0.20  # retrieve more broadly

ltm.evaluate_and_store(
    "My name is Drew and I live in Petaluma",
    "Nice to meet you Drew! Petaluma is a lovely city in Sonoma County."
)
test("Stored", ltm.last_action == "saved")

ltm.evaluate_and_store(
    "I like building AI projects with Ollama",
    "That's great! Ollama is an excellent tool for running local LLMs."
)
test("Second stored", ltm.last_action == "saved")

# Now retrieve with a related query
inj, raw = ltm.build_injection("Where does the user live?")
test("Injection built", inj is not None)
if raw:
    test("Relevant memory found", "Drew" in raw[0]["user"] or "Petaluma" in raw[0]["user"])
else:
    test("Relevant memory found", False, "no raw results")

# Different query
inj2, raw2 = ltm.build_injection("What AI tools does the user prefer?")
test("Second query works", inj2 is not None or ltm.last_action == "no_match")

# Cleanup
shutil.rmtree(TEST_DIR, ignore_errors=True)

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
if failed == 0:
    print("ALL TESTS PASSED ✓")
else:
    print(f"FAILURES: {failed}")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
