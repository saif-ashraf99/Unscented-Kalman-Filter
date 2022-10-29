#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
#include <iostream>

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initially set to false]
    is_initialized_ = false;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 3;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 1;

    /**
     * Measurement noise values provided by the sensor manufacturer.
     */

     // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    // State dimension
    n_x_ = 5;

    // Augmented state dimension
    n_aug_ = n_x_ + 2;

    // Sigma point spreading parameter
    lambda_ = 3 - n_x_;

    // Weights of sigma points
    weights_ = VectorXd(2 * n_aug_ + 1);

    // matrix to hold sigma points
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    // mean predicted measurement
    VectorXd z_pred;

    // measurement covariance matrix
    MatrixXd S;

    // current NIS for radar
    float NIS_radar_ = 0.0;

    // current NIS for laser
    float NIS_laser_ = 0.0;

    //measurement noise covariance matrix
    MatrixXd R;

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    if (!is_initialized_) {

        if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

            float x = meas_package.raw_measurements_(0);
            float y = meas_package.raw_measurements_(1);

            x_ << x, y, 0, 0, 0;
            P_ << std_laspx_ * std_laspx_, 0, 0, 0, 0,
                0, std_laspy_* std_laspy_, 0, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 0, 1;
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

            double rho = meas_package.raw_measurements_[0];
            double phi = meas_package.raw_measurements_[1];
            double rho_dot = meas_package.raw_measurements_[2];

            x_ << rho * cos(phi), rho* sin(phi), 0, 0, 0;
            P_ << std_radr_ * std_radr_, 0, 0, 0, 0,
                0, std_radr_* std_radr_, 0, 0, 0,
                0, 0, std_radrd_* std_radrd_, 0, 0,
                0, 0, 0, std_radphi_* std_radphi_, 0,
                0, 0, 0, 0, 1;
        }
        else
        {
            std::cout << "Unknown Sensor Type" << std::endl;
        }

        is_initialized_ = true;
        time_us_ = meas_package.timestamp_;
    }

    // delta_t expressed in seconds
    float delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;

    // Prediction Step
    Prediction(delta_t);

    // Update Step
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
        UpdateLidar(meas_package);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR) {
        UpdateRadar(meas_package);
    }
    else {
        std::cout << "Unknown Sensor Type" << std::endl;
    }

}


MatrixXd UKF::GenerateSigmaPoints()
{
    // create augmented mean vector
    VectorXd x_aug = VectorXd(7);

    // create augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);

    // create augmented state covariance
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    // create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    // create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;

    // create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    // create augmented sigma points
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; ++i) {
        Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }

    return Xsig_aug;
}

// Predict new Sigma Points by applying each augmented sigma point to nonlinear CTRV mapping function
void UKF::PredictSigmaPoints(MatrixXd& Xsig_aug, double delta_t)
{
    MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

    // predict sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {

        // extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        // predicted state values
        double px_p, py_p;

        // avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        }
        else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        // add Process Noise component
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a * delta_t;

        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;

        // write predicted sigma point into right column
        Xsig_pred(0, i) = px_p;
        Xsig_pred(1, i) = py_p;
        Xsig_pred(2, i) = v_p;
        Xsig_pred(3, i) = yaw_p;
        Xsig_pred(4, i) = yawd_p;

    }

    Xsig_pred_ = Xsig_pred;
}

void UKF::PredictMeanAndCovariance() {
    // create vector for predicted state
    VectorXd x = VectorXd(n_x_);

    // create covariance matrix for prediction
    MatrixXd P = MatrixXd(n_x_, n_x_);

    // predicted state mean
    // predicted state mean
    x.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
        x = x + weights_(i) * Xsig_pred_.col(i);
    }

    // predicted state covariance matrix
    P.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
      // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x;
        // angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        P = P + weights_(i) * x_diff * x_diff.transpose();
    }

    // Update State Vector X_ and Covariance P_
    x_ = x;
    P_ = P;
}

void UKF::Prediction(double delta_t) {
    /*
     Estimate the object's location. Modify the state vector, x_.
     Predict sigma points, the state, and the state covariance matrix.
    */

    // Generate Sigma Points
    MatrixXd Xsig_aug = GenerateSigmaPoints();

    // Convert Sigma Points through nonlinear mapping function
    PredictSigmaPoints(Xsig_aug, delta_t);

    //Predict state mean and covariance matrix
    PredictMeanAndCovariance();
}

void UKF::TransformSigmaPointsIntoMeasurementSpaceForLidar(int& n_z) {
    Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        Zsig(0, i) = Xsig_pred_(0, i);
        Zsig(1, i) = Xsig_pred_(1, i);
    }
}

void UKF::PredictMeasurementAndCovarianceForLidar(VectorXd& z_pred, MatrixXd& S, int& n_z) {
    // mean predicted measurement
    z_pred = VectorXd(n_z);

    // measurement covariance matrix
    S = MatrixXd(n_z, n_z);

    // Compute Predicted Measurement Mean
    z_pred.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    // Compute Predicted Measurement Covariance Matrix
    S.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

}

void UKF::AddMeasurementNoiseCovarianceMatrixForLidar(MatrixXd& R, int& n_z) {

    // measurement noise covariance matrix
    R = MatrixXd(n_z, n_z);

    R << std_laspx_ * std_laspx_, 0,
        0, std_laspy_* std_laspy_;

    S = S + R;
}


void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
     * Use lidar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the lidar NIS, if desired.
     */

    int n_z = 2;
    TransformSigmaPointsIntoMeasurementSpaceForLidar(n_z);

    PredictMeasurementAndCovarianceForLidar(z_pred, S, n_z);

    AddMeasurementNoiseCovarianceMatrixForLidar(R, n_z);

    MatrixXd Tc = MatrixXd(n_x_, n_z);

    Tc.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        // angle normalisation
        while (x_diff(3) > M_PI) {
            x_diff(3) -= 2.0 * M_PI;
        }
        while (x_diff(3) < -M_PI) {
            x_diff(3) += 2.0 * M_PI;
        }

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    // lidar measurement
    VectorXd z = meas_package.raw_measurements_;

    // residual
    VectorXd z_diff = z - z_pred;

    // update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    // calculate NIS
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

}

void UKF::TransformSigmaPointsIntoMeasurementSpaceForRadar(int& n_z) {
    Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    Zsig.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // rho, phi, rho_dot
        Zsig(0, i) = sqrt(px * px + py * py);
        Zsig(1, i) = atan2(py, px);
        Zsig(2, i) = (px * v1 + py * v2) / sqrt(px * px + py * py);
    }
}

void UKF::PredictMeasurementAndCovarianceForRadar(VectorXd& z_pred, MatrixXd& S) {
    int n_z = 3;
    z_pred = VectorXd(n_z);
    z_pred.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }
    S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // angle normalisation
        while (z_diff(1) > M_PI) {
            z_diff(1) -= 2. * M_PI;
        }
        while (z_diff(1) < -M_PI) {
            z_diff(1) += 2. * M_PI;
        }

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
}

void UKF::AddMeasurementNoiseCovarianceMatrixForRadar(MatrixXd& R) {
    int n_z = 3;
    R = MatrixXd(n_z, n_z);
    R << std_radr_ * std_radr_, 0, 0,
        0, std_radphi_* std_radphi_, 0,
        0, 0, std_radrd_* std_radrd_;

    S = S + R;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
     * Use radar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the radar NIS, if desired.
     */
    int n_z = 3;
    TransformSigmaPointsIntoMeasurementSpaceForRadar(n_z);

    PredictMeasurementAndCovarianceForRadar(z_pred, S);

    AddMeasurementNoiseCovarianceMatrixForRadar(R);

    MatrixXd Tc = MatrixXd(n_x_, n_z);

    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // angle normalisation
        while (z_diff(1) > M_PI) {
            z_diff(1) -= 2.0 * M_PI;
        }
        while (z_diff(1) < -M_PI) {
            z_diff(1) += 2.0 * M_PI;
        }

        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        // angle normalisation
        while (x_diff(3) > M_PI) {
            x_diff(3) -= 2.0 * M_PI;
        }
        while (x_diff(3) < -M_PI) {
            x_diff(3) += 2.0 * M_PI;
        }

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    // kalman gain;
    MatrixXd K = Tc * S.inverse();

    // radar measurement
    VectorXd z = meas_package.raw_measurements_;

    // residual
    VectorXd z_diff = z - z_pred;

    // angle normalisation
    while (z_diff(1) > M_PI) {
        z_diff(1) -= 2.0 * M_PI;
    }
    while (z_diff(1) < -M_PI) {
        z_diff(1) += 2.0 * M_PI;
    }

    // update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    // calculate NIS
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

}