#pragma once
/**
 * thread_pool.hpp
 * ===============
 * Minimal parallel_for using std::thread.
 *
 * Splits a range [0, n) into chunks and runs each chunk on a separate
 * thread.  No dependencies, no external libraries.
 */

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

namespace threading {

/**
 * Get the default number of threads.
 * Returns hardware_concurrency(), clamped to [1, 256].
 */
inline int default_threads() {
    int n = static_cast<int>(std::thread::hardware_concurrency());
    return std::max(1, std::min(n, 256));
}

/**
 * parallel_for(0, n, n_threads, func)
 *
 * Calls func(thread_id, start, end) for each chunk.
 * If n_threads <= 1, runs single-threaded (no std::thread overhead).
 */
inline void parallel_for(
    int n,
    int n_threads,
    const std::function<void(int thread_id, int start, int end)>& func
) {
    if (n <= 0) return;
    n_threads = std::max(1, std::min(n_threads, n));

    if (n_threads == 1) {
        func(0, 0, n);
        return;
    }

    int chunk = (n + n_threads - 1) / n_threads;
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        int start = t * chunk;
        int end   = std::min(start + chunk, n);
        if (start >= end) break;
        threads.emplace_back(func, t, start, end);
    }

    for (auto& th : threads) th.join();
}

} // namespace threading
