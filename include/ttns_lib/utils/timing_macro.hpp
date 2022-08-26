#ifndef TIMING_MACRO_HPP
#define TIMING_MACRO_HPP

#include <chrono>

/*
template <typename F, typename ... Ts>
std::pair<decltype(std::forward<F>(f)(std::forward<Ts>(args)...)), double> 
Time_Function(F&& f, Ts&& ... args)
{
    using func_ret_type = decltype(std::forward<F>(f)(std::forward<Ts>(args)...);
    try
    {
        auto start = std::chrono::high_resolution_clock::now();
        func_ret_type ret = std::forward<F>(f)(std::forward<Ts>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        
    }   
    catch(...){throw;}
}*/

#ifdef TIMING
#define INIT_TIMER auto start = std::chrono::high_resolution_clock::now();
#define START_TIMER  start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name)  std::cerr << "RUNTIME of " << name << ": " << \
    std::chrono::duration_cast<std::chrono::microseconds>( \
            std::chrono::high_resolution_clock::now()-start \
    ).count() << " us " << std::endl; 
#else
#define INIT_TIMER
#define START_TIMER
#define STOP_TIMER(name)
#endif

#ifdef TIMING_T
#define INIT_TIMER_T auto start_T = std::chrono::high_resolution_clock::now();
#define START_TIMER_T  start_T = std::chrono::high_resolution_clock::now();
#define STOP_TIMER_T(name)  std::cerr << "RUNTIME of " << name << ": " << \
    std::chrono::duration_cast<std::chrono::microseconds>( \
            std::chrono::high_resolution_clock::now()-start_T \
    ).count() << " us " << std::endl; 
#else
#define INIT_TIMER_T
#define START_TIMER_T
#define STOP_TIMER_T(name)
#endif

#endif

