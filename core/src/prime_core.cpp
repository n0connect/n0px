/**
 * @file prime_core.cpp
 * @brief Prime-X Core v5.3 (Reactive Miner)
 * @details Features: Reactive worker selection, optimized monitor loop, consolidated stats
 */

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <cstring>
#include <cmath>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <zmq.hpp>
#include <gmpxx.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/err.h>
#include <array>
#include <cstdint>

#include "prime_config.h" 

// --- COLORS ---
#define RESET   "\033[0m"
#define RED     "\033[38;5;196m"   
#define GREEN   "\033[38;5;46m"    
#define YELLOW  "\033[38;5;226m"   
#define BLUE    "\033[38;5;39m"    
#define PURPLE  "\033[38;5;129m"   
#define CYAN    "\033[38;5;51m"    
#define GRAY    "\033[38;5;240m"   
#define BOLD    "\033[1m"

// --- CONSTANTS ---

static constexpr size_t HASH_SIZE = 32;
static constexpr size_t SUPPORT_VEC_SIZE = HASH_SIZE * 3;
static constexpr size_t TRAILER_SIZE = 112; // PXSV + ver + rsv + seq + 3*hash


static inline void write_u64_le(uint8_t* dst, uint64_t v) {
    for (int i = 0; i < 8; ++i) dst[i] = (uint8_t)((v >> (8*i)) & 0xFF);
}

static inline std::array<uint8_t, HASH_SIZE> blake2s256(const uint8_t* data, size_t len) {
    std::array<uint8_t, HASH_SIZE> out{};
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) throw std::runtime_error("EVP_MD_CTX_new failed");

    if (EVP_DigestInit_ex(ctx, EVP_blake2s256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("EVP_DigestInit_ex(blake2s256) failed");
    }
    if (EVP_DigestUpdate(ctx, data, len) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("EVP_DigestUpdate failed");
    }
    unsigned int out_len = 0;
    if (EVP_DigestFinal_ex(ctx, out.data(), &out_len) != 1 || out_len != HASH_SIZE) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("EVP_DigestFinal_ex failed");
    }
    EVP_MD_CTX_free(ctx);
    return out;
}


// --- ENUMS ---
enum DataType {
    TYPE_EASY_COMPOSITE = 0,
    TYPE_PRIME = 1,
    TYPE_HARD_COMPOSITE = 2,
    TYPE_DP_PRIME = 3,
    TYPE_DP_COMPOSITE = 4,
    TYPE_DP_HARD_COMPOSITE = 5
};

// --- SYNC ---
std::mutex g_mutex;          
std::condition_variable g_cv;

// Go'dan gelen emirler (Varsayılan: AÇIK)
bool g_run_p = true;
bool g_run_h = true;
bool g_run_e = true;

std::atomic<bool> g_running(true);
std::atomic<bool> g_debug(false);

// Stats
std::atomic<uint64_t> s_p(0), s_h(0), s_e(0);

void signal_handler(int) { g_running = false; g_cv.notify_all(); }

struct DataBlock { mpz_class value; DataType type; };

// --- THREAD-SAFE QUEUE ---
class SafeQueue {
    std::queue<DataBlock> q; std::mutex m; std::condition_variable c;
    const size_t MAX = Config::INTERNAL_QUEUE_MAX;
public:
    void push(const mpz_class& v, DataType t) {
        std::unique_lock<std::mutex> l(m);
        // Sistem çalışıyorsa ve yer varsa bas, yoksa bekle
        c.wait(l, [this]{ return q.size() < MAX || !g_running; });
        if(!g_running) return;
        q.push({v,t}); c.notify_one();
    }
    bool pop(DataBlock& out) {
        std::unique_lock<std::mutex> l(m);
        if(c.wait_for(l, std::chrono::milliseconds(10), [this]{ return !q.empty() || !g_running; })) {
            if(q.empty()) return false;  // System shutting down
            out = q.front(); q.pop(); c.notify_all(); return true;
        }
        return false;
    }
    void drain() {
        std::lock_guard<std::mutex> l(m);
        while(!q.empty()) q.pop();
        c.notify_all();
    }
    size_t size() { std::lock_guard<std::mutex> l(m); return q.size(); }
};
SafeQueue g_queue;

// --- CRYPTO (THREAD-SAFE) ---
std::mutex g_rng_mutex;  // Global RNG protection
class SecureRNG {
    EVP_CIPHER_CTX* ctx; uint8_t k[32], iv[16]; std::vector<uint8_t> buf; size_t pos;
    void refill() { int l; std::vector<uint8_t> z(4096,0); EVP_EncryptUpdate(ctx, buf.data(), &l, z.data(), 4096); buf.resize(l); pos=0; }
public:
    SecureRNG() : buf(4096), pos(4096) { ctx=EVP_CIPHER_CTX_new(); if(!ctx) throw std::runtime_error("EVP_CIPHER_CTX_new failed"); RAND_bytes(k,32); RAND_bytes(iv,16); EVP_EncryptInit_ex(ctx,EVP_chacha20(),nullptr,k,iv); }
    ~SecureRNG() { if(ctx) EVP_CIPHER_CTX_free(ctx); }
    void bytes(uint8_t* d, size_t s) { std::lock_guard<std::mutex> lock(g_rng_mutex); if(pos+s>buf.size()) refill(); if(pos+s>buf.size()) throw std::runtime_error("RNG refill insufficient"); memcpy(d,&buf[pos],s); pos+=s; }
    uint32_t u32() { uint32_t v; bytes((uint8_t*)&v,4); return v; }
    float gauss() { 
        // Box-Muller with proper normalization to [1e-6, 1.0) to avoid log(0)
        // FIX: Removed +1.0f bias, using proper float division
        float u1 = (float)u32() / 4294967296.0f;
        float u2 = (float)u32() / 4294967296.0f;
        // Clamp u1 to avoid log(0) or log(1)
        if(u1 < 1e-6f) u1 = 1e-6f;
        if(u1 > 0.9999999f) u1 = 0.9999999f;
        return (std::sqrt(-2.0f*std::log(u1))*std::cos(6.28318f*u2))*Config::NOISE_SIGMA;
    }
};

// --- GENERATORS ---
mpz_class gen_p(SecureRNG& r) {
    mpz_class n; std::vector<uint8_t> b(Config::RAW_SIZE);
    while(1) { r.bytes(b.data(),Config::RAW_SIZE); mpz_import(n.get_mpz_t(),Config::RAW_SIZE,1,1,1,0,b.data()); mpz_setbit(n.get_mpz_t(),0); mpz_setbit(n.get_mpz_t(),Config::BITS-1); if(mpz_probab_prime_p(n.get_mpz_t(),Config::MILLER_RABIN_ROUNDS)>0) return n; }
}
mpz_class gen_h(SecureRNG& r) {
    size_t hb=Config::RAW_SIZE/2; mpz_class p,q; std::vector<uint8_t> b(hb);
    do{r.bytes(b.data(),hb); mpz_import(p.get_mpz_t(),hb,1,1,1,0,b.data()); mpz_setbit(p.get_mpz_t(),0); mpz_setbit(p.get_mpz_t(),Config::BITS/2-1);}while(!mpz_probab_prime_p(p.get_mpz_t(),Config::MILLER_RABIN_ROUNDS));
    do{r.bytes(b.data(),hb); mpz_import(q.get_mpz_t(),hb,1,1,1,0,b.data()); mpz_setbit(q.get_mpz_t(),0); mpz_setbit(q.get_mpz_t(),Config::BITS/2-1);}while(p==q||!mpz_probab_prime_p(q.get_mpz_t(),Config::MILLER_RABIN_ROUNDS));
    return p*q;
}
mpz_class gen_e(SecureRNG& r) {
    mpz_class n; std::vector<uint8_t> b(Config::RAW_SIZE);
    while(1) { r.bytes(b.data(),Config::RAW_SIZE); mpz_import(n.get_mpz_t(),Config::RAW_SIZE,1,1,1,0,b.data()); mpz_setbit(n.get_mpz_t(),0); mpz_setbit(n.get_mpz_t(),Config::BITS-1); if(!mpz_probab_prime_p(n.get_mpz_t(),Config::MILLER_RABIN_ROUNDS)) return n; }
}

// --- WORKER ---
void worker([[maybe_unused]] int id) {
    SecureRNG rng;
    while(g_running) {
        bool do_p, do_h, do_e;
        {
            std::unique_lock<std::mutex> l(g_mutex);
            // Sadece hepsi kapalıysa bekle. Biri bile açıksa çalış.
            g_cv.wait(l, []{ return !g_running || (g_run_p || g_run_h || g_run_e); });
            if(!g_running) break;
            do_p=g_run_p; do_h=g_run_h; do_e=g_run_e;
        }

        // Type Seçimi: Uniform distribution using rejection sampling
        // FIX: Modulus bias eliminated via rejection sampling
        // This ensures mathematically perfect uniformity: each type has equal probability
        std::vector<int> opts;
        if(do_p) opts.push_back(1);  // PRIME
        if(do_e) opts.push_back(0);  // EASY_COMPOSITE
        if(do_h) opts.push_back(2);  // HARD_COMPOSITE
        if(opts.empty()) continue;

        // Rejection sampling: ensures uniform distribution
        // For n options, sample from [0, 2^32) and reject if >= n * floor(2^32/n)
        int type;
        if(opts.size() == 1) {
            type = opts[0];  // Only one option, no randomness needed
        } else if(opts.size() == 2) {
            // For 2 options: 2^32 % 2 = 0, so modulus is safe
            type = opts[rng.u32() % 2];
        } else {  // opts.size() == 3
            // For 3 options: use rejection sampling to avoid modulus bias
            // 2^32 / 3 = 1,431,655,765 -> use 3 * this as limit
            uint32_t limit = 4294967295u - (4294967295u % 3u);  // Largest multiple of 3 <= 2^32
            uint32_t r;
            do {
                r = rng.u32();
            } while (r >= limit);  // Rejection: retry if r in biased tail
            type = opts[r % 3];
        }
        
        // Generate number and push BOTH base + DP versions
        if(type==1) {
            // PRIME type
            mpz_class num = gen_p(rng);
            g_queue.push(num, TYPE_PRIME);           // Label 1
            g_queue.push(num, TYPE_DP_PRIME);        // Label 3 (same number, DP applied in sender)
        } else if(type==2) {
            // HARD_COMPOSITE type
            mpz_class num = gen_h(rng);
            g_queue.push(num, TYPE_HARD_COMPOSITE);  // Label 2
            g_queue.push(num, TYPE_DP_HARD_COMPOSITE); // Label 5 (same number, DP applied in sender)
        } else {
            // EASY_COMPOSITE type (type==0)
            mpz_class num = gen_e(rng);
            g_queue.push(num, TYPE_EASY_COMPOSITE);  // Label 0
            g_queue.push(num, TYPE_DP_COMPOSITE);    // Label 4 (same number, DP applied in sender)
        }
    }
}

// --- MONITOR (Supervisor Listener) ---
void monitor() {
    zmq::context_t ctx(1); zmq::socket_t sub(ctx, ZMQ_SUB);
    std::string pub_addr = std::string("tcp://") + Config::ZMQ_HOST + ":" + std::to_string(Config::ZMQ_PORT_CORE_PUB);
    sub.connect(pub_addr); sub.set(zmq::sockopt::subscribe, "");
    // Set receive timeout: 1000ms for graceful shutdown
    sub.set(zmq::sockopt::rcvtimeo, 1000);
    
    while(g_running) {
        zmq::message_t msg;
        try {
            // Blocking receive with timeout (more efficient than non-blocking)
            if(sub.recv(msg, zmq::recv_flags::none)) {
                std::string cmd(static_cast<char*>(msg.data()), msg.size());
                if(cmd.find("STATE:")==0) {
                    // Parse format: "STATE:p|h|e:pbuf|hbuf|ebuf"
                    // Example: "STATE:1|1|0:45|32|95"
                    int p = -1, h = -1, e = -1;
                    int pbuf = -1, hbuf = -1, ebuf = -1;
                    
                    // Parse first part: p|h|e (binary states)
                    size_t pos = 6; // After "STATE:"
                    if(pos < cmd.size() && (cmd[pos] == '0' || cmd[pos] == '1')) p = cmd[pos] - '0';
                    pos = cmd.find('|', pos);
                    if(pos != std::string::npos && pos+1 < cmd.size() && (cmd[pos+1] == '0' || cmd[pos+1] == '1')) h = cmd[pos+1] - '0';
                    pos = cmd.find('|', pos+1);
                    if(pos != std::string::npos && pos+1 < cmd.size() && (cmd[pos+1] == '0' || cmd[pos+1] == '1')) e = cmd[pos+1] - '0';
                    
                    // Parse second part: buffer percentages (after ':')
                    size_t colon_pos = cmd.find(':');
                    if(colon_pos != std::string::npos && colon_pos+1 < cmd.size()) {
                        // Parse pbuf
                        size_t pbuf_pos = colon_pos + 1;
                        size_t pipe1 = cmd.find('|', pbuf_pos);
                        if(pipe1 != std::string::npos) {
                            try {
                                pbuf = std::stoi(cmd.substr(pbuf_pos, pipe1 - pbuf_pos));
                            } catch(...) { pbuf = -1; }
                        }
                        
                        // Parse hbuf
                        if(pipe1 != std::string::npos && pipe1+1 < cmd.size()) {
                            size_t pipe2 = cmd.find('|', pipe1+1);
                            if(pipe2 != std::string::npos) {
                                try {
                                    hbuf = std::stoi(cmd.substr(pipe1+1, pipe2 - pipe1 - 1));
                                } catch(...) { hbuf = -1; }
                            }
                        }
                        
                        // Parse ebuf
                        if(pipe1 != std::string::npos && pipe1+1 < cmd.size()) {
                            size_t pipe2 = cmd.find('|', pipe1+1);
                            if(pipe2 != std::string::npos && pipe2+1 < cmd.size()) {
                                try {
                                    ebuf = std::stoi(cmd.substr(pipe2+1));
                                } catch(...) { ebuf = -1; }
                            }
                        }
                    }
                    
                    // Apply state changes only if valid
                    if(p >= 0 && h >= 0 && e >= 0) {
                        {
                            std::lock_guard<std::mutex> l(g_mutex);
                            // Binary state: 1=OPEN, 0=FULL
                            g_run_p=(p==1); g_run_h=(h==1); g_run_e=(e==1);
                        }
                        g_cv.notify_all(); // State değişti, işçileri uyandır
                    }
                    
                    if(g_debug && (pbuf >= 0 || hbuf >= 0 || ebuf >= 0)) {
                        std::cerr << "[BUFFER] P:" << pbuf << "% H:" << hbuf << "% E:" << ebuf << "%" << std::endl;
                    }
                }
            }
        } catch (const zmq::error_t& e) {
            // Timeout is expected, continue
            if(e.num() != EAGAIN) {
                if(g_debug) std::cerr << "Monitor recv error: " << e.what() << std::endl;
            }
        }
        
        // Heartbeat: Eğer açık emir varsa ve kuyruk boşalıyorsa dürt
        bool active; { std::lock_guard<std::mutex> l(g_mutex); active=(g_run_p||g_run_h||g_run_e); }
        if(active && g_queue.size() < 1900) g_cv.notify_all();
    }
}

// --- SENDER ---
void sender() {
    zmq::context_t ctx(1); zmq::socket_t sock(ctx, ZMQ_PUSH);
    std::string push_addr = std::string("tcp://*:") + std::to_string(Config::ZMQ_PORT_CORE_PUSH);
    sock.bind(push_addr);
    SecureRNG rng; 
    // FORMAT (v1 integrity trailer): [type:4][raw:RAW_SIZE][floats:BITS*4][trailer:112] = PACKET_SIZE
    // Packet size is dynamically calculated from Config (generated by configure.py from config.json)
    const size_t base_size = 4 + Config::RAW_SIZE + (Config::BITS * 4);
    const size_t pkt_size  = base_size + TRAILER_SIZE;
    if (pkt_size != (size_t)Config::PACKET_SIZE) {
        throw std::runtime_error("Packet size mismatch: expected " + std::to_string(Config::PACKET_SIZE) + 
                                 " but calculated " + std::to_string(pkt_size));
    }
    std::vector<uint8_t> pkt(pkt_size);
    std::atomic<uint64_t> seq{0};
    uint64_t pkt_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    std::cout << CYAN << BOLD << "╔══════════════════════════════════════════════════════╗" << RESET << "\n";
    std::cout << CYAN << BOLD << "║  PRIME-X CORE SENDER v5.3 - LIVE DATA STREAMING      ║" << RESET << "\n";
    std::cout << CYAN << BOLD << "╚══════════════════════════════════════════════════════╝" << RESET << "\n\n";

    while(g_running) {
        DataBlock b; if(!g_queue.pop(b)) continue;
        
        // Clear packet
        std::fill(pkt.begin(), pkt.end(), 0);
        
        // Write TYPE header at offset 0 (little-endian int32 for Go compatibility)
        int32_t type_val = (int32_t)b.type;
        pkt[0] = (type_val) & 0xFF;
        pkt[1] = (type_val >> 8) & 0xFF;
        pkt[2] = (type_val >> 16) & 0xFF;
        pkt[3] = (type_val >> 24) & 0xFF;
        
        // Export RAW integer (BIG-ENDIAN, network byte order) at offset 4
        // Raw starts at byte 4 and occupies Config::RAW_SIZE bytes
        size_t raw_start = 4;
        std::fill(pkt.data() + raw_start, pkt.data() + raw_start + Config::RAW_SIZE, 0);
        size_t cnt;
        mpz_export(pkt.data() + raw_start, &cnt, 1, 1, 1, 0, b.value.get_mpz_t());
        // If mpz_export wrote fewer bytes than RAW_SIZE, zero-pad at beginning (big-endian aligned)
        if (cnt < (size_t)Config::RAW_SIZE) {
            // Shift bytes to right: move cnt bytes to offset (RAW_SIZE - cnt)
            memmove(pkt.data() + raw_start + (Config::RAW_SIZE - cnt), pkt.data() + raw_start, cnt);
            std::fill(pkt.data() + raw_start, pkt.data() + raw_start + (Config::RAW_SIZE - cnt), 0);
        }
        
        // Float vector starts at offset 4 + Config::RAW_SIZE
        // Stores Config::BITS float32 values (one per bit, little-endian IEEE754)
        uint8_t* fp_base = pkt.data() + 4 + Config::RAW_SIZE;
        for(int i = 0; i < Config::BITS; ++i) {
            float noisy_bit = (float)mpz_tstbit(b.value.get_mpz_t(), i) + rng.gauss();
            memcpy(fp_base + (i*4), &noisy_bit, 4);  // Store as little-endian float32
        }
        
        // === INTEGRITY TRAILER (v1) ===
        const size_t off_type  = 0;
        const size_t off_raw   = 4;
        const size_t off_vec   = 4 + Config::RAW_SIZE;
        const size_t vec_bytes = (size_t)Config::BITS * 4;
        const size_t off_tr    = base_size;
        
        const uint64_t s = seq.fetch_add(1, std::memory_order_relaxed);
        
        // hashes
        auto h_raw = blake2s256(pkt.data() + off_raw, (size_t)Config::RAW_SIZE);
        auto h_vec = blake2s256(pkt.data() + off_vec, vec_bytes);
        
        // all_hash context = type||raw||vec||seq||raw_size||bits
        uint8_t seq_le[8]; 
        write_u64_le(seq_le, s);
        uint8_t meta[8];
        uint32_t raw_sz = (uint32_t)Config::RAW_SIZE;
        uint32_t bits   = (uint32_t)Config::BITS;
        // raw_sz LE
        meta[0]=raw_sz&0xFF; meta[1]=(raw_sz>>8)&0xFF; meta[2]=(raw_sz>>16)&0xFF; meta[3]=(raw_sz>>24)&0xFF;
        // bits LE
        meta[4]=bits&0xFF; meta[5]=(bits>>8)&0xFF; meta[6]=(bits>>16)&0xFF; meta[7]=(bits>>24)&0xFF;
        
        // all_hash incremental (avoid allocating giant buffer)
        EVP_MD_CTX* all = EVP_MD_CTX_new();
        if (!all) throw std::runtime_error("EVP_MD_CTX_new failed");
        if (EVP_DigestInit_ex(all, EVP_blake2s256(), nullptr) != 1) throw std::runtime_error("all init failed");
        EVP_DigestUpdate(all, pkt.data() + off_type, 4);
        EVP_DigestUpdate(all, pkt.data() + off_raw, (size_t)Config::RAW_SIZE);
        EVP_DigestUpdate(all, pkt.data() + off_vec, vec_bytes);
        EVP_DigestUpdate(all, seq_le, 8);
        EVP_DigestUpdate(all, meta, 8);
        std::array<uint8_t, HASH_SIZE> h_all{};
        unsigned int out_len=0;
        if (EVP_DigestFinal_ex(all, h_all.data(), &out_len) != 1 || out_len != HASH_SIZE) {
            EVP_MD_CTX_free(all);
            throw std::runtime_error("EVP_DigestFinal_ex failed");
        }
        EVP_MD_CTX_free(all);
        
        // trailer write
        pkt[off_tr + 0] = 'P'; pkt[off_tr + 1] = 'X'; pkt[off_tr + 2] = 'S'; pkt[off_tr + 3] = 'V';
        pkt[off_tr + 4] = 1; // version
        pkt[off_tr + 5] = 0; pkt[off_tr + 6] = 0; pkt[off_tr + 7] = 0; // reserved
        write_u64_le(pkt.data() + off_tr + 8, s);
        
        // support vector = raw||vec||all
        std::memcpy(pkt.data() + off_tr + 16,                h_raw.data(), HASH_SIZE);
        std::memcpy(pkt.data() + off_tr + 16 + HASH_SIZE,    h_vec.data(), HASH_SIZE);
        std::memcpy(pkt.data() + off_tr + 16 + 2*HASH_SIZE,  h_all.data(), HASH_SIZE);
        
        try { 
            sock.send(zmq::buffer(pkt), zmq::send_flags::none);
            int32_t t = (int32_t)b.type;
            if(t == 1 || t == 3) s_p++;        // Prime + DP_Prime
            else if(t == 2 || t == 5) s_h++;   // Hard Composite + DP_Hard
            else if(t == 0 || t == 4) s_e++;   // Easy Composite + DP_Easy
            
            pkt_count++;
            
            // Show progress every 100 packets
            if (pkt_count % 100 == 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                double rate = (elapsed > 0) ? (pkt_count * 1000.0 / elapsed) : 0;
                
                // Clear screen every 10000 packets to prevent terminal bloat
                if(pkt_count % 10000 == 0) {
                    std::cout << "\033[2J\033[H" << std::flush;  // Clear screen, move cursor to home
                }
                
                // Progress bar (50 chars)
                int bar_width = 40;
                int filled = (pkt_count % 1000) / (1000 / bar_width);
                
                std::cout << "\r" << GREEN << "✓ TX " << RESET 
                          << std::setw(6) << pkt_count << " packets | "
                          << BLUE << "[";
                for(int i = 0; i < bar_width; ++i) 
                    std::cout << (i < filled ? "█" : "░");
                std::cout << "]" << RESET << " "
                          << YELLOW << std::fixed << std::setprecision(1) << rate << " pkt/s" << RESET
                          << " | P:" << GREEN << s_p << RESET
                          << " H:" << RED << s_h << RESET
                          << " E:" << PURPLE << s_e << RESET
                          << "   " << std::flush;
            }
        } catch(...) {}
    }
    
    // Final stats
    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    double total_rate = (total_elapsed > 0) ? (pkt_count / (double)total_elapsed) : 0;
    
    std::cout << "\n\n" << CYAN << BOLD << "═══════════════════════════════════════════════════════" << RESET << "\n";
    std::cout << GREEN << BOLD << "✓ SENDER SHUTDOWN" << RESET << "\n";
    std::cout << "  Total Packets: " << BOLD << pkt_count << RESET << "\n";
    std::cout << "  Runtime: " << BOLD << total_elapsed << "s" << RESET << "\n";
    std::cout << "  Average Rate: " << BOLD << std::fixed << std::setprecision(2) << total_rate << " pkt/s" << RESET << "\n";
    std::cout << "  Prime (P): " << GREEN << s_p << RESET << " | "
              << "Hard Composite (H): " << RED << s_h << RESET << " | "
              << "Easy Composite (E): " << PURPLE << s_e << RESET << "\n";
    std::cout << CYAN << BOLD << "═══════════════════════════════════════════════════════" << RESET << "\n";
}

// --- MAIN ---
int main() {
    signal(SIGINT, signal_handler);
    std::cout << "\033[2J\033[H" << BOLD << "PRIME-X CORE v5.3 [REACTIVE MINER]" << RESET << "\n";
    
    std::vector<std::thread> p;
    p.emplace_back(sender);
    p.emplace_back(monitor);
    
    int wc = std::thread::hardware_concurrency()-2; if(wc<2) wc=2;
    for(int i=0; i<wc; ++i) p.emplace_back(worker, i);

    auto t0 = std::chrono::steady_clock::now(); uint64_t tot0=0;
    while(g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // UI güncellemesi
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()/1000.0;
        uint64_t P=s_p, H=s_h, E=s_e, TOT=P+H+E;
        
        std::string st;
        { std::lock_guard<std::mutex> l(g_mutex); st = (g_run_p?"P:ON ":"P:OFF ") + std::string(g_run_h?"H:ON ":"H:OFF ") + (g_run_e?"E:ON":"E:OFF"); }
        
        std::cout << "\r" << BLUE << "[STATUS] " << RESET << st << " | Speed: " << YELLOW << (uint64_t)((TOT-tot0)/dt) << " op/s" << RESET << " | Buf: " << g_queue.size() << " | P:"<<P<<" H:"<<H<<" E:"<<E << "   " << std::flush;
        tot0=TOT; t0=t1;
    }
    
    // Graceful shutdown: drain queue ve thread'leri bekle
    g_queue.drain();
    for(auto& t:p) {
        if(t.joinable()) t.join();
    }
    EVP_cleanup();
    
    std::cout << "\n" << GREEN << "[SHUTDOWN] All threads terminated cleanly." << RESET << std::endl;
    return 0;
}