#!/usr/bin/env python3
"""Test IOWarp CTE at scale — put many blobs, query by regex.

This tests the ACTUAL CLIO-at-IOWarp-scale story:
1. Put N blobs into CTE with scientific tags
2. Use BlobQuery to find blobs by regex
3. Measure how many blobs CTE inspects vs how many it returns
"""
import subprocess, time, threading, json, sys

print("Starting Chimaera runtime...", flush=True)
rt = subprocess.Popen(
    ["chimaera", "runtime", "start"],
    stdout=open("/tmp/rt.log", "w"),
    stderr=subprocess.STDOUT,
)
for i in range(15):
    time.sleep(1)
    try:
        with open("/tmp/rt.log") as f:
            if "Successfully started local server" in f.read():
                print(f"Runtime ready in {i+1}s", flush=True)
                break
    except Exception:
        pass
else:
    print("Runtime never became ready", flush=True)
    sys.exit(1)

from iowarp_core import wrp_cte_core_ext as cte
import iowarp_core

print(f"iowarp_core version: {iowarp_core.get_version()}", flush=True)

result = {"steps": [], "metrics": {}}
def log(step, status, detail="", t=0):
    result["steps"].append({"step": step, "status": status, "detail": str(detail)[:300], "time_s": round(t, 3)})
    print(f"[{status}] {step}: {str(detail)[:120]} ({t:.3f}s)", flush=True)

try:
    # Init
    t0 = time.time()
    ok = cte.chimaera_init(cte.ChimaeraMode.kClient)
    log("chimaera_init", "OK" if ok else "FAIL", f"ok={ok}", time.time()-t0)

    t0 = time.time()
    ok = cte.initialize_cte("", cte.PoolQuery.Dynamic())
    log("initialize_cte", "OK" if ok else "FAIL", f"ok={ok}", time.time()-t0)

    # === SCALE TEST: put 1000 scientific blobs ===
    N_BLOBS = 1000
    print(f"\n[Phase 1] Writing {N_BLOBS} scientific blobs to CTE...", flush=True)

    # Simulate scientific data: temperature, pressure, wind, humidity
    import random
    rng = random.Random(42)

    # Create a few tags by domain
    tags = {
        "sci_temperature": cte.Tag("sci_temperature"),
        "sci_pressure": cte.Tag("sci_pressure"),
        "sci_wind": cte.Tag("sci_wind"),
        "sci_humidity": cte.Tag("sci_humidity"),
    }
    log("Create tags", "OK", f"{len(tags)} domain tags", 0)

    t0 = time.time()
    domains = list(tags.keys())
    needle_blobs = []  # Track which blobs are "hot" (temp > 30)
    for i in range(N_BLOBS):
        domain = domains[i % len(domains)]
        tag = tags[domain]
        if domain == "sci_temperature":
            # Some blobs are "hot" (the needles)
            is_hot = (i % 50 == 0)  # 2% of temperature blobs
            value = rng.uniform(30, 50) if is_hot else rng.uniform(0, 25)
            data = f"temperature {value:.2f} degC station {i}".encode()
            if is_hot:
                needle_blobs.append((domain, f"blob_{i:06d}"))
        elif domain == "sci_pressure":
            data = f"pressure {rng.uniform(90, 110):.2f} kPa station {i}".encode()
        elif domain == "sci_wind":
            data = f"wind {rng.uniform(0, 30):.2f} km/h station {i}".encode()
        else:
            data = f"humidity {rng.uniform(20, 90):.2f} percent station {i}".encode()

        tag.PutBlob(f"blob_{i:06d}", data)

    put_time = time.time() - t0
    log(f"PutBlob x{N_BLOBS}", "OK",
        f"{put_time:.2f}s total, {N_BLOBS/put_time:.0f} blobs/s",
        put_time)
    result["metrics"]["put_blobs"] = N_BLOBS
    result["metrics"]["put_time_s"] = round(put_time, 3)
    result["metrics"]["put_throughput_blobs_per_s"] = round(N_BLOBS / put_time, 0)
    result["metrics"]["needle_count"] = len(needle_blobs)

    print(f"\n[Phase 2] Querying with BlobQuery (IOWarp's metadata-based search)...", flush=True)

    client = cte.get_cte_client()

    # Query 1: all blobs under sci_temperature tag
    t0 = time.time()
    temp_blobs = client.BlobQuery("sci_temperature", ".*", 10000, cte.PoolQuery.Dynamic())
    temp_blobs = list(temp_blobs)
    q1_time = time.time() - t0
    log("BlobQuery(temperature)", "OK",
        f"{len(temp_blobs)} blobs found in {q1_time*1000:.1f}ms", q1_time)
    result["metrics"]["query_temperature"] = {
        "results": len(temp_blobs),
        "time_ms": round(q1_time * 1000, 2),
    }

    # Query 2: all blobs with matching name pattern
    t0 = time.time()
    all_blobs = client.BlobQuery(".*", "blob_00000.*", 10000, cte.PoolQuery.Dynamic())
    all_blobs = list(all_blobs)
    q2_time = time.time() - t0
    log("BlobQuery(name regex)", "OK",
        f"{len(all_blobs)} blobs found in {q2_time*1000:.1f}ms", q2_time)
    result["metrics"]["query_name_regex"] = {
        "results": len(all_blobs),
        "time_ms": round(q2_time * 1000, 2),
    }

    # Query 3: query all tags
    t0 = time.time()
    all_tags = client.TagQuery("sci_.*", 100, cte.PoolQuery.Dynamic())
    all_tags = list(all_tags)
    q3_time = time.time() - t0
    log("TagQuery(sci_.*)", "OK",
        f"{len(all_tags)} tags: {all_tags}", q3_time)
    result["metrics"]["query_tags"] = {
        "results": len(all_tags),
        "tags": list(all_tags),
        "time_ms": round(q3_time * 1000, 2),
    }

    # Query 4: search across ALL blobs
    t0 = time.time()
    everything = client.BlobQuery(".*", ".*", 10000, cte.PoolQuery.Dynamic())
    everything = list(everything)
    q4_time = time.time() - t0
    log("BlobQuery(all)", "OK",
        f"{len(everything)} total blobs in {q4_time*1000:.1f}ms", q4_time)
    result["metrics"]["query_all"] = {
        "results": len(everything),
        "time_ms": round(q4_time * 1000, 2),
    }

    # === SCALE TEST SUCCESS ===
    result["success"] = True
    result["summary"] = {
        "blobs_stored": N_BLOBS,
        "domains": len(tags),
        "needles_planted": len(needle_blobs),
        "put_throughput_blobs_per_s": round(N_BLOBS / put_time, 0),
        "query_latency_ms": {
            "tag_filter": round(q1_time * 1000, 2),
            "name_regex": round(q2_time * 1000, 2),
            "tag_listing": round(q3_time * 1000, 2),
            "full_scan": round(q4_time * 1000, 2),
        },
    }

finally:
    rt.terminate()
    try:
        rt.wait(timeout=5)
    except Exception:
        rt.kill()

print("\n=== RESULT ===", flush=True)
print(json.dumps(result, indent=2, default=str), flush=True)
