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
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;
    default_random_engine re{};
    normal_distribution<double> dx{x, std[0]};
    normal_distribution<double> dy{y, std[1]};
    normal_distribution<double> dt{theta, std[2]};
    for (size_t i_particle = 0; i_particle < num_particles; i_particle++){
        Particle particle;
        particle.x = dx(re);
        particle.y = dy(re);
        particle.theta = dt(re);
        particle.id = i_particle;
        particle.weight = 1;
        
        particles.push_back(particle);
        weights.push_back(1);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine re{};

    for (size_t i_particle = 0; i_particle < num_particles; i_particle++){
        double new_x;
        double new_y;
        double new_theta;
        
        if (yaw_rate == 0){ // equations from lesson 12: part 3
            new_x = particles[i_particle].x + velocity*delta_t*cos(particles[i_particle].theta);
            new_y = particles[i_particle].y + velocity*delta_t*sin(particles[i_particle].theta);
            new_theta = particles[i_particle].theta;
        }
        else
        {
            new_theta = particles[i_particle].theta + yaw_rate*delta_t;
            new_x = particles[i_particle].x + velocity/yaw_rate*(sin(new_theta)-sin(particles[i_particle].theta));
            new_y = particles[i_particle].y + velocity/yaw_rate*(cos(particles[i_particle].theta) - cos(new_theta));
        }
       
        normal_distribution<double> dx{new_x, std_pos[0]};
        normal_distribution<double> dy{new_y, std_pos[1]};
        normal_distribution<double> dt{new_theta, std_pos[2]};
        
        particles[i_particle].x = dx(re);
        particles[i_particle].y = dy(re);
        particles[i_particle].theta = dt(re);
        
    }
    

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, double sensor_range) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (size_t i_obs = 0; i_obs < observations.size(); i_obs++) {
        //Maximum distance can be square root of 2 times the range of sensor.
        double lowest_dist = sensor_range * sqrt(2);
        int closest_landmark_id = -1;

        for (size_t i_land = 0; i_land < predicted.size(); i_land++) {
            double current_dist = dist(observations[i_obs].x, observations[i_obs].y, predicted[i_land].x, predicted[i_land].y);

            if (current_dist < lowest_dist) {
                lowest_dist = current_dist;
                closest_landmark_id = predicted[i_land].id;
            }
        }
        observations[i_obs].id = closest_landmark_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    double weight_normalizer = 0.0;
    
    for (size_t i_particle = 0; i_particle < num_particles; i_particle++) {
        
        vector<LandmarkObs> trans_obs; // equation from Lesson 14 section 15 for transformation matrix from vehicle to map coordinates
        for (size_t i_obs = 0; i_obs < observations.size(); i_obs++ ){
            LandmarkObs obs;
            obs.id = i_obs;
            obs.x = cos(particles[i_particle].theta)*observations[i_obs].x - sin(particles[i_particle].theta)*observations[i_obs].y + particles[i_particle].x;
            obs.y = sin(particles[i_particle].theta)*observations[i_obs].x + cos(particles[i_particle].theta)*observations[i_obs].y + particles[i_particle].y;
            trans_obs.push_back(obs);
        }
        
        //filter map landmarks, only keep within sensor range.
        vector<LandmarkObs> predicted_landmarks;
        for (size_t i_mapL = 0; i_mapL < map_landmarks.landmark_list.size(); i_mapL++) {
            Map::single_landmark_s current_landmark = map_landmarks.landmark_list[i_mapL];
            if ((fabs((particles[i_particle].x - current_landmark.x_f)) <= sensor_range) && (fabs((particles[i_particle].y - current_landmark.y_f)) <= sensor_range)) {
                //predicted_landmarks.push_back(current_landmark);
                predicted_landmarks.push_back(LandmarkObs {current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
            }
        }
        
        //make the data association between landmark and observations
        dataAssociation(predicted_landmarks, trans_obs, sensor_range);
        
        // update weights
        particles[i_particle].weight = 1.0;
        double sigma_x = std_landmark[0];
        double sigma_y = std_landmark[1];
        double normalizer = (1.0/(2.0 * M_PI * sigma_x * sigma_y));
        
        for (size_t i_obs = 0; i_obs < trans_obs.size(); i_obs++){
            double multi_prob = 1.0;
            for (size_t i_land = 0; i_land < predicted_landmarks.size(); i_land++){
                if(trans_obs[i_obs].id == predicted_landmarks[i_land].id){ // equation from lesson 14, section 19
                    multi_prob = normalizer*exp(-1.0*((pow(trans_obs[i_obs].x-predicted_landmarks[i_land].x,2)/(2.0*pow(sigma_x,2)))+(pow(trans_obs[i_obs].y-predicted_landmarks[i_land].y,2)/(2.0*pow(sigma_y,2)))));
                    particles[i_particle].weight *= multi_prob;
                }
            }
        }
        weight_normalizer += particles[i_particle].weight;
    }
    
    //normalize weight with overall weight for overall probability equal to 1
    for (size_t i_particle = 0; i_particle < num_particles; i_particle++) {
        particles[i_particle].weight /= weight_normalizer;
        weights[i_particle] = particles[i_particle].weight;
    }
  
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine re{};
    discrete_distribution<int> distribution(weights.begin(), weights.end());
    
    vector<Particle> resampled_particles; 
    for (size_t i_particle = 0; i_particle < num_particles; i_particle++) {
        resampled_particles.push_back(particles[distribution(re)]);
    }
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
