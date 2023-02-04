#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "neuron.hpp"

class neural_network {
public:
	neural_network(const std::vector<size_t>& topology);
	~neural_network();

	void feed_forward(const std::vector<double>& input);
	void back_propagation(const std::vector<double>& target);
	void get_results(std::vector<double>& output) const;

	double get_recent_average_error() const;

//private:
	std::vector<std::vector<neuron>> layers_;

	double error_;
	double recent_average_error_;
	constexpr static double recent_average_smoothing_factor_ = 100.0;
};

#endif