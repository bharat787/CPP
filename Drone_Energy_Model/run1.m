% load a list of waypoints (in meters)
A = csvread('wpts.csv');
waypoints = A;
% load the original energy model (2016)
load('energy_model/E_model.mat')
% the improvement presented at IROS2018 is inside the struct E_model:
% E_model.E_angle_speed , computed by Taua Cabreira and Carmelo Di Franco


% decide if to used the improvement or not. note that this depends on the
% autopilot. In our experiment we do not stop at every waypoints, hence we
% also modeled the speed reduction at every turn. If you want to simplify
% the problem you can stop at every waypoint (final speed = 0)
stop_at_every_waypoint = true;
% you can decide if to compute the optimal speed or fly at a constant
% speed. if the distance to cover is not enough to accelerate and
% decelerate the speed will not reach the maximum speed 
compute_optimal_speed = true;

% altitute (if you want to include the climbing/descending phase
altitude = 20;
% if you want to have a maximum speed (if the optimal speed is greater then
% it will consider the maximum speed. If you want to consider the optimal
% speed then put max_speed = 14;
max_speed = 8;
[total_energy, total_time, total_distance] = predict_energy_mission...
    (E_model,waypoints, altitude,max_speed, stop_at_every_waypoint);
%plot area of interest
title('Coverage path planning scenario')
plot(waypoints(:,1),waypoints(:,2),'b');
axis equal;
xlabel('x [m]')
xlabel('y [m]')
% print results
fprintf('Total Energy is %f J \n', total_energy);
fprintf('Total time is %f s \n',total_time);
fprintf('Total distance is %f m \n',total_distance);


   
   
