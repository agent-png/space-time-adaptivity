#include "Timing.hpp"

void Profiler::tic(const std::string& name)
{
    timers[name].tic();
}

void Profiler::toc(const std::string& name)
{
    timers[name].toc();
}

void Profiler::report(){
    std::cout << "---------------------------------------------------------------\n";
        std::cout << std::left
                  << std::setw(18) << "Timer"
                  << std::setw(10) << "Calls"
                  << std::setw(10) << "Min(s)"
                  << std::setw(10) << "Max(s)"
                  << std::setw(10) << "Total(s)"
                  << "\n";
        std::cout << "---------------------------------------------------------------\n";
    for(auto timer : timers){
        std::cout << std::left
                      << std::setw(18) << timer.first
                      << std::setw(10) << timer.second.calls
                      << std::setw(10) << timer.second.min
                      << std::setw(10) << timer.second.max 
                      << std::setw(10) << timer.second.total
                      << "\n";
    } 
     
}