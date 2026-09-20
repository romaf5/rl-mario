// Persistent worker pool: parallel_for(n, f, chunk) runs f(i, worker) for i in
// [0, n) with dynamic chunks; the calling thread is worker 0.
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ss {

class Pool {
public:
    explicit Pool(int threads) : n_(threads < 1 ? 1 : threads) {
        for (int w = 1; w < n_; w++) th_.emplace_back([this, w] { work(w); });
    }
    ~Pool() {
        { std::lock_guard<std::mutex> l(m_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    int size() const { return n_; }
    void parallel_for(int64_t n, const std::function<void(int64_t, int)>& f, int64_t chunk = 1) {
        if (n <= 0) return;
        {
            std::lock_guard<std::mutex> l(m_);
            f_ = &f; items_ = n; chunk_ = chunk < 1 ? 1 : chunk; next_ = 0; busy_ = n_ - 1; gen_++;
        }
        cv_.notify_all();
        run(0);
        std::unique_lock<std::mutex> l(m_);
        done_.wait(l, [&] { return busy_ == 0; });
    }

private:
    void run(int w) {
        for (;;) {
            int64_t i = next_.fetch_add(chunk_);
            if (i >= items_) return;
            const int64_t e = std::min(i + chunk_, items_);
            for (; i < e; i++) (*f_)(i, w);
        }
    }
    void work(int w) {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> l(m_);
                cv_.wait(l, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
            }
            run(w);
            std::lock_guard<std::mutex> l(m_);
            if (--busy_ == 0) done_.notify_one();
        }
    }
    int n_;
    std::vector<std::thread> th_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    uint64_t gen_ = 0;
    int busy_ = 0;
    bool stop_ = false;
    const std::function<void(int64_t, int)>* f_ = nullptr;
    std::atomic<int64_t> next_{0};
    int64_t items_ = 0, chunk_ = 1;
};

}  // namespace ss
