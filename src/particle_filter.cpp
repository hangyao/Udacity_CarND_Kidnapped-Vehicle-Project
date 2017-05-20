/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>

#include "particle_filter.h"
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	default_random_engine gen;
	normal_distribution<double> x_distribution(x, std[0]);
	normal_distribution<double> y_distribution(y, std[1]);
	normal_distribution<double> theta_distribution(theta, std[2]);
	num_particles = 1000;
	for (unsigned int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.x = x_distribution(gen);
		particle.y = y_distribution(gen);
		particle.theta = theta_distribution(gen);
		particle.weight = 1;
		particles.push_back(particle);
	}
	weights.resize(num_particles);
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	default_random_engine gen;
	normal_distribution<double> x_noise(0, std_pos[0]);
	normal_distribution<double> y_noise(0, std_pos[1]);
	normal_distribution<double> theta_noise(0, std_pos[2]);
	for (unsigned int i = 0; i < num_particles; i++) {
		Particle particle = particles[i];
    float dyaw_dt     = yaw_rate * delta_t;
		float theta_prime = particle.theta + dyaw_dt;
		float v_by_dyaw   = velocity / yaw_rate;
		particle.x += v_by_dyaw * (sin(theta_prime) - sin(particle.theta)) + x_noise(gen);
		particle.y += v_by_dyaw * (cos(particle.theta) - cos(theta_prime)) + y_noise(gen);
		particle.theta = theta_prime + theta_noise(gen);
		particles[i] = particle;
	}
}

vector<LandmarkObs> ParticleFilter::transformLandmarks(Particle particle, Map map_landmarks) {
	vector<LandmarkObs> transformed_landmarks;
	for (unsigned int i = 0; i < map_landmarks.landmark_list.size(); i++) {
		Map::single_landmark_s landmark = map_landmarks.landmark_list[i];
		LandmarkObs transformed_landmark;
		transformed_landmark.id = landmark.id_i;
		double cos_theta = cos(particle.theta - M_PI / 2);
		double sin_theta = sin(particle.theta - M_PI / 2);
		transformed_landmark.x = -(landmark.x_f - particle.x) * sin_theta + (landmark.y_f - particle.y) * cos_theta;
		transformed_landmark.y = -(landmark.x_f - particle.x) * cos_theta - (landmark.y_f - particle.y) * sin_theta;
		transformed_landmarks.push_back(transformed_landmark);
	}
	return transformed_landmarks;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
  for (unsigned int i = 0; i < num_particles; i++) {
		Particle particle = particles[i];
			vector<LandmarkObs> transformed_landmarks = transformLandmarks(particle, map_landmarks);
			sort(transformed_landmarks.begin(), transformed_landmarks.end());
			double posterior_obs = 1.0;
			for (unsigned int j = 0; j < observations.size(); j++) {
				LandmarkObs prediction = transformed_landmarks[j];
				LandmarkObs observation = observations[j];
				double prediction_point[] = {prediction.x, prediction.y};
				double observed_point[] = {observation.x, observation.y};
				posterior_obs *= bivariate_gausian(observed_point, prediction_point, std_landmark);
			}
			particle.weight = posterior_obs;
			particles[i] = particle;
			weights[i] = posterior_obs;
		}
}

void ParticleFilter::resample() {
	default_random_engine gen;
	double max_weight = *max_element(begin(weights), end(weights));
	uniform_real_distribution<double> beta_uniform_distribution(0, 2 * max_weight);
  double beta = 0.0;
	unsigned int index = (rand() % (int)(num_particles + 1));
	vector<Particle> resampled_particles;
	for (unsigned int i = 0; i < num_particles; i++) {
		beta += beta_uniform_distribution(gen);
		while (weights[index] < beta) {
			beta = beta - weights[index];
			index = (index + 1) % num_particles;
		}
		Particle sample = particles[index];
		Particle particle;
		particle.x = sample.x;
		particle.y = sample.y;
		particle.theta = sample.theta;
		particle.weight = sample.weight;
		resampled_particles.push_back(particle);
	}
	particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
