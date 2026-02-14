# Query-Aware Pruning: Why SPTAG Uses Sequential vs Batch Loading

## The Key Insight

SPTAG's query-aware pruning works because they use **ASYNC I/O** with **BATCH_READ**, not sequential blocking I/O!

## SPTAG's Architecture

### 1. Async Batch Read (SPTAG's Approach)
```cpp
// From ExtraStaticSearcher.h line 350
#ifdef BATCH_READ
    BatchReadFileAsync(m_indexFiles, (p_exWorkSpace->m_diskRequests).data(), postingListCount);
#endif
```

**How it works:**
1. **Prepare all requests** - Loop through posting IDs and prepare async read requests
2. **Submit batch** - `BatchReadFileAsync()` submits ALL requests to OS at once
3. **OS parallelizes** - Operating system schedules disk I/O in parallel
4. **Process as ready** - Results processed as they complete
5. **Early termination** - Can stop when `m_loadedPostingNum >= m_searchInternalResultNum`

**Key advantage:** OS can reorder I/O requests for optimal disk access patterns!

### 2. Query-Aware Pruning Logic
```cpp
// From SPANNIndex.cpp line 633
if (ret == ErrorCode::VectorNotFound && 
    extraWorkspace->m_loadedPostingNum >= m_options.m_searchInternalResultNum)
    extraWorkspace->m_relaxedMono = true;
```

**Pruning happens AFTER batch submission:**
- All I/O requests submitted upfront
- Processing stops early when enough results found
- Remaining I/O requests can be cancelled or ignored

## Our Implementation (Why It Failed)

### Sequential Blocking I/O
```python
for centroid_id in nearest_centroids:
    result = self._load_posting_mmap(centroid_id)  # BLOCKS here!
    # Process result
    if loaded_count >= search_internal_result_num:
        break  # Too late - already waited for I/O
```

**Problem:**
1. Each `_load_posting_mmap()` **blocks** waiting for disk
2. Early termination only saves **future** I/O, not current
3. No parallelization - disk sits idle between requests
4. Result: **3.5× slower** than batch loading

### Our Batch Loading (Current)
```python
# Load ALL postings at once
postings = self._load_postings_batch(list(nearest_centroids))

# Then search them
for centroid_id in nearest_centroids:
    # Process posting
```

**Why it's faster:**
- Single mmap operation loads contiguous data
- OS can prefetch and cache efficiently
- No per-posting overhead
- Result: **2.4× speedup** from single-file format

## The Solution: Async Batch Read

To make query-aware pruning work, we need:

### Option 1: Python asyncio with aiofiles
```python
import asyncio
import aiofiles

async def load_postings_async(self, centroid_ids):
    tasks = []
    for cid in centroid_ids:
        task = asyncio.create_task(self._load_posting_async(cid))
        tasks.append(task)
    
    # Submit all at once
    results = await asyncio.gather(*tasks)
    
    # Process with early termination
    loaded = 0
    for result in results:
        if loaded >= self.search_internal_result_num:
            break  # Can stop early
        # Process result
        loaded += 1
```

### Option 2: Thread Pool (Simpler)
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def search_with_pruning(self, query, centroids):
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all I/O at once
        futures = {
            executor.submit(self._load_posting_mmap, cid): cid 
            for cid in centroids
        }
        
        loaded = 0
        for future in as_completed(futures):
            if loaded >= self.search_internal_result_num:
                break  # Early termination works!
            
            result = future.result()
            # Process result
            loaded += 1
```

### Option 3: io_uring (Linux, Best Performance)
```python
import liburing  # Requires Linux 5.1+

# Submit all reads to io_uring
ring = liburing.io_uring()
for cid in centroids:
    ring.prep_read(fd, buffer, offset)
ring.submit()

# Process completions with early termination
loaded = 0
for completion in ring.completions():
    if loaded >= search_internal_result_num:
        break
    # Process result
    loaded += 1
```

## Performance Comparison

| Approach | Latency | Why |
|----------|---------|-----|
| Sequential (our attempt) | 43ms | Blocks on each I/O |
| Batch sync (current) | 12.4ms | Single mmap, OS prefetch |
| Async batch (SPTAG) | ~5-7ms | Parallel I/O + early stop |

## Why SPTAG is Faster

1. **Parallel I/O** - Multiple disk requests in flight
2. **OS optimization** - Kernel reorders for disk efficiency  
3. **Early termination** - Stop processing, not I/O submission
4. **NVMe benefits** - High queue depth utilizes parallelism

## Recommendation

For our Python implementation:

**Short term:** Keep batch loading (current approach)
- Simple, fast enough (12.4ms)
- Hit 90% recall target ✅

**Long term:** Add async I/O option
- Use ThreadPoolExecutor for simplicity
- Enable with flag: `use_async_io=True`
- Expected: 2× speedup on NVMe

**On EC2 NVMe:**
- Current batch: ~5-7ms (2-3× faster than macOS)
- With async: ~3-4ms (would hit <10ms target easily)

## Key Takeaway

**Query-aware pruning requires async I/O to be effective!**

Without async:
- Sequential = slow (blocks on each I/O)
- Batch = fast (single operation, OS prefetch)

With async:
- Batch async = fastest (parallel I/O + early termination)
