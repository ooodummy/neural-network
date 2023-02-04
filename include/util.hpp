#ifndef UTIL_HPP
#define UTIL_HPP

#include <random>

namespace util {
	template <typename T>
	T random(T min, T max) {
		static std::random_device rd;
		static std::mt19937 gen(rd());

		if constexpr (!std::is_integral_v<T>) {
			std::uniform_real_distribution<T> dist(min, max);
			return dist(gen);
		} else {
			std::uniform_int_distribution<T> dist(min, max);
			return dist(gen);
		}
	}
}

#endif