#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"]; // global x-position of waypoints
          vector<double> ptsy = j[1]["ptsy"]; // global y-position of waypoints
          double px = j[1]["x"];              // global x-position of vehicle
          double py = j[1]["y"];              // global y-position of vehicle
          double psi = j[1]["psi"];           // orientation of vehicle [radians]
          double v = j[1]["speed"];           // speed of vehicle [mph]
          double throttle = j[1]["throttle"];
          double steering_angle = j[1]["steering_angle"];

          // First, transform the global waypoints to vehicle coordinates
          for (int i=0; i<ptsx.size(); i++) {
            // Shift the waypoint coordinates to be centered about the current vehicle origin
            double x_shifted = ptsx[i] - px;
            double y_shifted = ptsy[i] - py;

            // Then, rotate the variables into the current vehicle reference frame.
            // Just overwrite the existing values
            double angle = -psi;

            // Cache results of expensive trig operations.
            double cosAng = cos(angle);
            double sinAng = sin(angle);

            ptsx[i] = x_shifted * cosAng - y_shifted * sinAng;
            ptsy[i] = x_shifted * sinAng + y_shifted * cosAng;
          }

          // Now, compute the crosstrack error (cte) and the heading error (epsi).
          // To do this, we need to fit a cubic polynomial to the local points, 
          // which need to be expressed as Eigen::VectorXd objects.

          // Copy over the elements of the vector<double> objects into VectorXd objects.
          Eigen::VectorXd x_glob_wp(ptsx.size());
          Eigen::VectorXd y_glob_wp(ptsy.size());
          for (int i=0; i<ptsx.size(); i++) {
            x_glob_wp[i] = ptsx[i];
            y_glob_wp[i] = ptsy[i];
          }

          /** NOTE TO REVIEWER: Polynomial Fitting and MPC Preprocessing

          Here is where the polynomial is fit to the waypoints.  The 
          waypoints have been transformed from global coordinates into 
          a vehicle-centered reference frame.
        
          */
          // Fit the cubic polynomial to the points.
          Eigen::VectorXd fit = polyfit(x_glob_wp, y_glob_wp, 3);

          // Now, evaluate the cross track error (cte) relative to this fit cubic polynomial.
          double cte = polyeval(fit, 0);
          std::cout << "Crosstrack Error (CTE) = " << cte << " [m] " << std::endl;

          double epsi = psi - atan(fit[1]);
          
          
          /* NOTE TO REVIEWER: Model Predictve Control with Latency

            To account for latency, the state is predicted forward by the 
            latency delay (100 ms) so the solver outputs a command that is
            time-synched with time of actuation by the effectors.

            Note the prediction kinematics have been slighly modified (simplified) 
            to account for the fact that initial states (x,y,psi) are defined to
             be zero post-alignment with the vehicle-centered reference frame.

          */
          double latency_dt = 0.1; // 100 ms
          double Lf = 2.67;


          double latency_x = v * latency_dt;
          double latency_y = 0;
          double latency_psi = -(v / Lf) * steering_angle * latency_dt;
          double latency_v = v + throttle * latency_dt;
          double latency_cte = cte + v * sin(epsi) * latency_dt;

          // Compute the expected heading based on fit.
          double expected_psi = atan(fit[1] + 
                                2.0 * fit[2] * latency_x + 
                                3.0 * fit[3] * latency_x*latency_x);

          // Compute the latent heading error.
          double latency_epsi = psi - expected_psi;
          
          // Compose the state to pass into the solver.
          Eigen::VectorXd state(6); // State has 6 elements
          state << latency_x, latency_y, latency_psi, latency_v, latency_cte, latency_epsi;

          /*
          * Calculate steeering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */

          // Solve for the steering angle and acceleration solution.
          vector<double> solution = mpc.Solve(state, fit);

          // Assign the solution outputs to clarifying variable names.
          double steer_value = solution[0];
          double throttle_value = solution[1];

          // Package the JSON payload for transmission to the simulator.
          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory 
          msgJson["mpc_x"] = mpc.traj_x;
          msgJson["mpc_y"] = mpc.traj_y;

          //Display the waypoints/reference line
          msgJson["next_x"] = ptsx;
          msgJson["next_y"] = ptsy;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
