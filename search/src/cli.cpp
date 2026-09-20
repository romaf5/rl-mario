// smbsearch CLI over the C API: benchmarks and the PGO profile workload.
//   smbsearch bench    ROM STATE [threads] [frames]
//   smbsearch explore  ROM STATE ROUTE BUDGET_S OUT [threads]
//   smbsearch optimize ROM STATE ROUTE REF OUT [beam] [threads]
// STATE: a raw (not gzipped) native savestate; ROUTE like 1-1,1-2,4-1,4-2,8-1,8-2,8-3,8-4;
// REF / OUT: action files (one byte per action).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "../include/smbsearch.h"

static std::vector<uint8_t> slurp(const char* path) {
    std::vector<uint8_t> d;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    uint8_t b[65536];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) d.insert(d.end(), b, b + n);
    fclose(f);
    return d;
}

static std::vector<int32_t> parse_route(const char* s) {
    std::vector<int32_t> r;
    std::string t(s);
    size_t i = 0;
    while (i < t.size()) {
        const int w = t[i] - '0', l = t[i + 2] - '0';
        r.push_back((w - 1) * 4 + (l - 1));
        i += 4;
    }
    return r;
}

static void dump(const char* path, const std::vector<uint8_t>& a, int n) {
    FILE* f = fopen(path, "wb");
    fwrite(a.data(), 1, (size_t)n, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: see cli.cpp\n"); return 2; }
    const std::string cmd = argv[1];
    std::vector<uint8_t> rom = slurp(argv[2]), state = slurp(argv[3]);
    if ((int)state.size() != ss_state_size()) { fprintf(stderr, "state size %zu != %d\n", state.size(), ss_state_size()); return 2; }
    if (cmd == "bench") {
        ss_ctx* c = ss_create(rom.data(), (int)rom.size(), argc > 4 ? atoi(argv[4]) : 0);
        const long long frames = argc > 5 ? atoll(argv[5]) : 4000000LL;
        printf("%.0f frames/s on %d threads\n", ss_bench(c, state.data(), frames), ss_threads(c));
        ss_destroy(c);
        return 0;
    }
    std::vector<int32_t> route = parse_route(argv[4]);
    std::vector<uint8_t> out(200000);
    ss_stats st{};
    if (cmd == "explore" && argc >= 7) {
        ss_ctx* c = ss_create(rom.data(), (int)rom.size(), argc > 7 ? atoi(argv[7]) : 0);
        const int n = ss_explore(c, state.data(), route.data(), (int)route.size(), atof(argv[5]), 10.0, 300, 1,
                                 out.data(), (int)out.size(), &st);
        printf("explore: found %d, %d actions, %lld cells, %lld walks, %.1f s\n", st.found, n,
               (long long)st.cells, (long long)st.walks, st.seconds);
        if (n > 0) dump(argv[6], out, n);
        ss_destroy(c);
        return st.found ? 0 : 1;
    }
    if (cmd == "optimize" && argc >= 7) {
        ss_ctx* c = ss_create(rom.data(), (int)rom.size(), argc > 8 ? atoi(argv[8]) : 0);
        std::vector<uint8_t> ref = slurp(argv[5]);
        const int n = ss_optimize(c, state.data(), route.data(), (int)route.size(), ref.data(), (int)ref.size(),
                                  argc > 7 ? atoi(argv[7]) : 20000, 16, 6000, 1, out.data(), (int)out.size(), &st);
        printf("optimize: found %d, %d actions (reference %zu), %.1f s\n", st.found, n, ref.size(), st.seconds);
        if (n > 0) dump(argv[6], out, n);
        ss_destroy(c);
        return st.found ? 0 : 1;
    }
    fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
