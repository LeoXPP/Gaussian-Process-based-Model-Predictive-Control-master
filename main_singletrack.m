%------------------------------------------------------------------
% Programed by: 
%   - Lucas Rath (lucasrm25@gmail.com)
%   - 
%   -

%   Control of a Race Car in a Race Track using Gaussian Process Model Predictive Control:
%------------------------------------------------------------------

clear all; 


%--------------------------------------------------------------------------
% Quick Access Simulation and controller parameters
%------------------------------------------------------------------
dt = 0.15;       % simulation  timestep size
tf = 1.5*60;       % simulation time
maxiter = 20;   % max NMPC iterations per time step
N = 10;         % NMPC prediction horizon

loadPreTrainedGP =  true;
% GPfile = fullfile(pwd,'/simresults/20-01-15-out-GP-without-GP.mat');
GPfile = fullfile(pwd,'xpp_trai4littleweight.mat');
useGP = true;
trainGPonline = true;
useParallel = true;


% display info
lookahead = dt*N;
fprintf('\nPrediction lookahead: %.1f [s]\n',lookahead);



%% Create True Dynamics Simulation Model
%--------------------------------------------------------------------------
%   xk+1 = fd_true(xk,uk) + Bd * ( w ),    
%
%       where: w ~ N(0,var_w)
%------------------------------------------------------------------

% define noise for true disturbance
var_w = diag([(1/3)^2 (1/3)^2 (deg2rad(3)/3)^2]);
% var_w = zeros(3);

% create true dynamics model
trueModel = MotionModelGP_SingleTrack_true( [], var_w);
% trueModel = MotionModelGP_SingleTrack_nominal(d,var_w);


%% Create Estimation Model and Nominal Model

% -------------------------------------------------------------------------
%  Create nominal model (no disturbance):  
%       xk+1 = fd_nom(xk,uk)
% -------------------------------------------------------------------------

nomModel = MotionModelGP_SingleTrack_nominal( [], [] ); 
% nomModel = MotionModelGP_SingleTrack_true( [], [] );

nomModel.analyseSingleTrack();


% -------------------------------------------------------------------------
%  Create adaptive dynamics model 
%  (unmodeled dynamics will be estimated by Gaussian Process GP)
%       xk+1 = fd_nom(xk,uk) + Bd * ( d_GP(zk) + w )
% -------------------------------------------------------------------------

if ~loadPreTrainedGP
    % GP input dimension
    gp_n = MotionModelGP_SingleTrack_nominal.nz;
    % GP output dimension
    gp_p = MotionModelGP_SingleTrack_nominal.nd;

    % GP hyperparameters
    var_f   = repmat(0.01,[gp_p,1]);    % output variance
    var_n   = diag(var_w/3);              % measurement noise variance
    M       = repmat(diag([1e0,1e0,1e0,1e0,1e0].^2),[1,1,gp_p]);     % length scale
%     var_f   = repmat(0.001,[gp_p,1]);    % output variance
%     var_n   = diag(var_w/3);              % measurement noise variance
%     M       = repmat(diag([0.001,0.001,0.001,0.001,0.001].^2),[1,1,gp_p]);     % length scale
    maxsize = 300; % maximum number of points in the dictionary

    % create GP object
    d_GP = GP(gp_n, gp_p, var_f, var_n, M, maxsize);
else
    load(GPfile); %,'d_GP'
    fprintf('\nGP model loaded succesfuly\n\n')
end

% ---------------------------------------------------------------------
% create nominal model with GP model as d(zk)（ adaptive dynamics model
% ）--------------------------------------------------------------------
estModel = MotionModelGP_SingleTrack_nominal(@d_GP.eval, var_w);
% estModel = MotionModelGP_SingleTrack_true(@d_GP.eval, var_w);


%% Initialize Controller

% -------------------------------------------------------------------------
% Create perception model (in this case is the saved track points)
% this is needed to for the MPC cost function
% -------------------------------------------------------------------------
[trackdata, x0, th0, w] = RaceTrack.loadTrack_03();
track = RaceTrack(trackdata, x0, th0, w);
% TEST: [Xt, Yt, PSIt, Rt] = track.getTrackInfo(1000)
%       trackAnim = SingleTrackAnimation(track,mpc.N);
%       trackAnim.initGraphics()


% -------------------------------------------------------------------------
% Nonlinear Model Predictive Controller
% -------------------------------------------------------------------------

% define cost function
n  = estModel.n;
m  = estModel.m;
ne = 0;

% define cost functions
eig_XY_max =0.0;
shrink_ = 0.0;
fo   = @(t,mu_x,var_x,u,e,r) costFunction(mu_x, var_x, u, track,eig_XY_max,shrink_);            % e = track distance
fend = @(t,mu_x,var_x,e,r)   2 * costFunction(mu_x, var_x, zeros(m,1), track,eig_XY_max,shrink_);   % end cost function

% define dynamics 这里定义了系统动态，即如何从当前状态和控制输入预测下一个状态。
f  = @(mu_x,var_x,u) estModel.xkp1(mu_x, var_x, u, dt);
%f  = @(mu_x,var_x,u) trueModel.xkp1(mu_x, var_x, u, dt);
% define additional constraints
h  = @(x,u,e) [];
g  = @(x,u,e) [];
u_lb = [-deg2rad(20);  % >= steering angle
         -1;           % >= gas pedal
         5];           % >= centerline track velocity
u_ub = [deg2rad(20);   % <= steering angle
        1;             % <= gas pedal
        30];           % <= centerline track velocity 


% Initialize NMPC object;
% 最后，使用前面定义的所有参数和函数初始化 NMPC 对象，
% 并设置优化问题的容忍度（tol）和最大迭代次数（maxiter）。
mpc = NMPC(f, h, g, u_lb, u_ub, n, m, ne, fo, fend, N, dt);
mpc.tol     = 1e-2;
mpc.maxiter = maxiter;



%% Prepare simulation
% ---------------------------------------------------------------------
% Prepare simulation (initialize vectors, initial conditions and setup
% animation
% ---------------------------------------------------------------------

% define variable sizes
true_n = trueModel.n;
true_m = trueModel.m;
est_n = estModel.n;
est_m = estModel.m;

% initial state
x0 = [5;0;0; 10;0;0; 0];   % true initial state
x0(end) = track.getTrackDistance(x0(1:2)); % get initial track traveled distance

% change initial guess for mpc solver. Set initial track velocity as
% initial vehicle velocity (this improves convergence speed a lot)
%  MPC 求解器的初始猜测。将初始轨迹速度设置为初始车辆速度，这可以大大提高收敛速度。
mpc.uguess(end,:) = x0(4)*2;

% define simulation time
out.t = 0:dt:tf;            % time vector
kmax = length(out.t)-1;     % steps to simulate

% initialize variables to store simulation results
out.x              = [x0 NaN(true_n,kmax)];             % true states
out.xhat           = [x0 NaN(est_n, kmax)];             % state estimation
out.xnom           = [x0 NaN(est_n, kmax)];             % predicted nominal state
out.u              =     NaN(est_m, kmax);              % applied input
out.x_ref          = NaN(2,     mpc.N+1, kmax);         % optimized reference trajectory
out.mu_x_pred_opt  = NaN(mpc.n, mpc.N+1, kmax);         % mean of optimal state prediction sequence
out.var_x_pred_opt = NaN(mpc.n, mpc.n, mpc.N+1, kmax);  % variance of optimal state prediction sequence
out.u_pred_opt     = NaN(mpc.m, mpc.N,   kmax);         % open-loop optimal input prediction


% start animation
trackAnim = SingleTrackAnimation(track, out.mu_x_pred_opt, out.var_x_pred_opt, out.u_pred_opt, out.x_ref);
trackAnim.initTrackAnimation();
trackAnim.initScope();
drawnow;

% deactivate GP evaluation in the prediction
% d_GP.isActive = useGP;
d_GP.isActive = true;
fprintf('\nGP active? %s\n\n',string(useGP))



%% Start simulation

ki = 1;
% ki = 310;
% mpc.uguess = out.u_pred_opt(:,:,ki);


for k = ki:kmax
    disp('------------------------------------------------------')
    fprintf('time: %.3f [s]\n',out.t(k))
    
    if(k>2)
    % -------------------------------------------------------------------------
    % Nonlinear Model Predictive Controller
    % -------------------------------------------------------------------------
    load('eig_XY_max.mat');
    % define cost function
    n  = estModel.n;
    m  = estModel.m;
    ne = 0;
 
    % define cost functions
    fo   = @(t,mu_x,var_x,u,e,r) costFunction(mu_x, var_x, u, track,eig_XY_max,shrink_);            % e = track distance
    fend = @(t,mu_x,var_x,e,r)   2 * costFunction(mu_x, var_x, zeros(m,1), track,eig_XY_max,shrink_);   % end cost function
    
    % define dynamics 这里定义了系统动态，即如何从当前状态和控制输入预测下一个状态。
    f  = @(mu_x,var_x,u) estModel.xkp1(mu_x, var_x, u, dt);
    %f  = @(mu_x,var_x,u) trueModel.xkp1(mu_x, var_x, u, dt);
    % define additional constraints
    h  = @(x,u,e) [];
    g  = @(x,u,e) [];
    u_lb = [-deg2rad(20);  % >= steering angle
             -1;           % >= gas pedal
             5];           % >= centerline track velocity
    u_ub = [deg2rad(20);   % <= steering angle
            1;             % <= gas pedal
            30];           % <= centerline track velocity 


    % Initialize NMPC object;
    % 最后，使用前面定义的所有参数和函数初始化 NMPC 对象，
    % 并设置优化问题的容忍度（tol）和最大迭代次数（maxiter）。
    mpc = NMPC(f, h, g, u_lb, u_ub, n, m, ne, fo, fend, N, dt);
    mpc.tol     = 1e-2;
    mpc.maxiter = maxiter;
    mpc.uguess=u_opt(:,[2:end,end]);
    end
    % ---------------------------------------------------------------------
    % LQR controller
    % ---------------------------------------------------------------------
    % % out.u(:,i) = Kr*out.r(:,i) - K*out.xhat(:,i);
    
    % ---------------------------------------------------------------------
    % NPMC controller
    % ---------------------------------------------------------------------
    % calculate optimal input
    [u_opt, e_opt] = mpc.optimize(out.xhat(:,k), out.t(k), 0, useParallel);
    out.u(:,k) = u_opt(:,1);
    sprintf('\nSteering angle: %d\nTorque gain: %.1f\nTrack vel: %.1f\n',rad2deg(out.u(1,k)),out.u(2,k),out.u(3,k))

    % ---------------------------------------------------------------------
    % Calculate predicted trajectory from optimal open-loop input sequence 
    % and calculate optimized reference trajectory for each prediction
    % ---------------------------------------------------------------------
    % get optimal state predictions from optimal input and current state
    out.u_pred_opt(:,:,k) = u_opt;
    [out.mu_x_pred_opt(:,:,k),out.var_x_pred_opt(:,:,:,k)] = mpc.predictStateSequence(out.xhat(:,k), zeros(estModel.n), u_opt);
  
    var_XY =  out.var_x_pred_opt(1:2,1:2,5,k);
    eig_XY = eig(var_XY);
    eig_XY_max = max(eig_XY);
    eig_XY_max = abs(eig_XY_max);
    shrink_ = eig_XY_max;
    eig_XY_max = min(0.0, eig_XY_max);
    fprintf('eig_XY_max is : %f \n', eig_XY_max);
    save('eig_XY_max.mat','eig_XY_max');




    % get target track distances from predictions (last state)
    out.x_ref(:,:,k) = track.getTrackInfo(out.mu_x_pred_opt(end,:,k));
    
    % ---------------------------------------------------------------------
    % update race animation and scopes
    % ---------------------------------------------------------------------
    trackAnim.mu_x_pred_opt  = out.mu_x_pred_opt;
    trackAnim.var_x_pred_opt = out.var_x_pred_opt;
    trackAnim.u_pred_opt     = out.u_pred_opt;
    trackAnim.x_ref          = out.x_ref;
    trackAnim.updateTrackAnimation(k);
    trackAnim.updateScope(k);
    drawnow;
    
    % ---------------------------------------------------------------------
    % Simulate real model
    % ---------------------------------------------------------------------
    [mu_xkp1,var_xkp1] = trueModel.xkp1(out.x(:,k),zeros(trueModel.n),out.u(:,k),dt);
    % out.x(:,k+1) = mvnrnd(mu_xkp1, var_xkp1, 1)';
    out.x(:,k+1) = mu_xkp1;
    
    % ---------------------------------------------------------------------
    % Measure data
    % ---------------------------------------------------------------------
    out.xhat(:,k+1) = out.x(:,k+1); % perfect observer 即测量值就是真实模型的输出值
    % get traveled distance, given vehicle coordinates
    out.xhat(end,k+1) = track.getTrackDistance( out.xhat([1,2],k+1) , out.xhat(end,k) );
    
    
    % ---------------------------------------------------------------------
    % Lap timer
    % ---------------------------------------------------------------------
    [laptimes, idxnewlaps] = RaceTrack.getLapTimes(out.xhat(end,:),dt);
    if any(k==idxnewlaps)
        RaceTrack.dispLapTimes(laptimes);
    end
    
    
    % ---------------------------------------------------------------------
    % Safety - Stop simulation in case vehicle is completely unstable
    % ---------------------------------------------------------------------
    V_vx = out.xhat(4,k+1);
    V_vy = out.xhat(5,k+1);
    beta = atan2(V_vy,V_vx);
    if V_vx < 0
        fprintf('Vehicle is driving backwards... aborting');
        break;
    end
    if abs(rad2deg(beta)) > 80
        fprintf('Vehicle has a huge sideslip angle... aborting');
        break;
    end    
    
    % ---------------------------------------------------------------------
    % calculate nominal model
    % ---------------------------------------------------------------------
    out.xnom(:,k+1) = nomModel.xkp1(out.xhat(:,k),zeros(nomModel.n),out.u(:,k),dt);
    
    
    % ---------------------------------------------------------------------
    % Add data to GP model
    % ---------------------------------------------------------------------
    if mod(k-1,1)==0
        % calculate disturbance (error between measured and nominal)
        d_est = estModel.Bd \ (out.xhat(:,k+1) - out.xnom(:,k+1));
        % d_est = estModel.Bd \ (mu_xkp1 - out.xnom(:,k+1));
        % select subset of coordinates that will be used in GP prediction
        zhat = [ estModel.Bz_x * out.xhat(:,k); estModel.Bz_u * out.u(:,k) ];
        % add data point to the GP dictionary
        if trainGPonline
            d_GP.add(zhat,d_est');
            d_GP.updateModel();
        end
        
        fprintf('Prediction Error norm WITHOUT GP: %f\n',norm(d_est));
        disp(d_est)
        fprintf('Prediction Error norm WITH    GP: %f\n',norm(d_est-d_GP.eval(zhat,true)));
        % fprintf('Prediction Error norm WITH    GP: %f\n',norm(d_est-estModel.d(zhat,true)));
        disp(d_est-d_GP.eval(zhat,true))
    end
    
    % if length(laptimes) >= 6
        % d_GP.isActive = true;
        % mpc.maxiter = 30;
    % end

% xiepanpan for GP-UPDATE 
end


% Display Lap times

[laptimes, idxnewlaps] = RaceTrack.getLapTimes(out.xhat(end,:),dt);
RaceTrack.dispLapTimes(laptimes)



% xiepanpan for GP-UPDATE 
% return
% STOP here. Next sections are intended to be executed separately



%% Readd simulation data to GP, uddate model and optimize parameters

k = find(~isnan(out.xhat(1,:)), 1, 'last' ) - 20;

% create new instance of GP class
d_GP = GP(gp_n, gp_p, var_f, var_n, M, maxsize);

% readd points
d_est = estModel.Bd \ (out.xhat(:,2:k) - out.xnom(:,2:k));
zhat  = estModel.z( out.xhat(:,1:k-1), out.u(:,1:k-1) );
d_GP.add(zhat,d_est');

% update and optimize model
d_GP.updateModel();
d_GP.optimizeHyperParams('fmincon')

d_GP.M
d_GP.var_f
d_GP.var_n


%% Analyse learning
% ---------------------------------------------------------------------
% Check how the GP reduces the prediction error
% ---------------------------------------------------------------------
%%
% 
%   for x = 1:10
%       disp(x)
%   end
% 

% d_GP.optimizeHyperParams('fmincon')
% d_GP.optimizeHyperParams('ga')


k = find(~isnan(out.xhat(1,:)), 1, 'last' ) - 20;

% prediction error without GP
% predErrorNOgp = estModel.Bd\(out.xhat - out.xnom);
predErrorNOgp = estModel.Bd\(out.xhat(:,1:k-1) - out.xnom(:,1:k-1));


% prediction error with trained GP
zhat  = estModel.z( out.xhat(:,1:k-1), out.u(:,1:k-1) );
dgp = d_GP.eval(zhat,true);
predErrorWITHgp = estModel.Bd\( out.xhat(:,2:k) - (out.xnom(:,2:k) + estModel.Bd*dgp) );


disp('Prediction mean squared error without GP:')
disp( mean(predErrorNOgp(:,all(~isnan(predErrorNOgp))).^2 ,2) )
disp('Prediction mean squared error with trained GP:')
disp( mean(predErrorWITHgp(:,all(~isnan(predErrorWITHgp))).^2 ,2) )



% Visualize error
figure('Color','w'); hold on; grid on;
subplot(1,2,1)
plot( predErrorNOgp' )
subplot(1,2,2)
hist(predErrorNOgp')
sgtitle('Prediction error - without GP')


figure('Color','w'); hold on; grid on;
subplot(1,2,1)
plot( predErrorWITHgp' )
subplot(1,2,2)
hist(predErrorWITHgp')
sgtitle('Prediction error - with GP')


% ---------------------------------------------------------------------
% Check in which region of the tyre dynamics we are working
% ---------------------------------------------------------------------

% % % % simulation
% % % 
% trueModel.testTyres
% 
% l_f  = 0.9;
% l_r  = 1.5;
% V_vx = out.xhat(4,:);
% V_vy = out.xhat(5,:);
% psi_dot = out.xhat(6,:);
% delta = out.u(1,:);
% a_r = atan2(V_vy-l_r.*psi_dot,V_vx);
% a_f = atan2(V_vy+l_f.*psi_dot,V_vx) - [delta 0];
% 
% figure('Color','w'); hold on; grid on;
% plot(rad2deg(a_r))
% plot(rad2deg(a_f))
% ylabel('slip angle')
% xlabel('time step')

% xiepanpan for GP-UPDATE


%% Show animation
%close all;

% start animation
trackAnim = SingleTrackAnimation(track, out.mu_x_pred_opt, out.var_x_pred_opt, out.u_pred_opt, out.x_ref);
trackAnim.initTrackAnimation();
trackAnim.initScope();
for k=1:kmax
    if ~ trackAnim.updateTrackAnimation(k)
        break;
    end
    % trackAnim.updateScope(k);
%     pause(0.15);
    drawnow;
end


%% Record video

FrameRate = 7;
videoName = fullfile('simresults',sprintf('trackAnimVideo-%s',date));
videoFormat = 'Motion JPEG AVI';
trackAnim.recordvideo(videoName, videoFormat, FrameRate);


%% Cost function for the MPC controller

function cost = costFunction(mu_x, var_x, u, track,eig_XY_max,shrink_)

    % Track oriented penalization
%     q_l   = 50;     % penalization of lag error
%     q_c   = 50;     % penalization of contouring error
%     q_o   = 5;      % penalization for orientation error
%     q_d   = -3;     % reward high track centerline velocites
%     q_r   = 100;    % penalization when vehicle is outside track
    q_l   = 50;     % penalization of lag error
    q_c   = 50*1.3 + shrink_*0;     % penalization of contouring error
    q_o   = 5*1.2;      % penalization for orientation error
    q_d   = -3;     % reward high track centerline velocites
    q_r   = 100*0.8;    % penalization when vehicle is outside track
    
    % state and input penalization
%     q_v      = -0;  % reward high absolute velocities
%     q_st     =  0;  % penalization of steering
%     q_br     =  0;  % penalization of breaking
%     q_psidot =  8;  % penalize high yaw rates
%     q_acc    = -0;  % reward for accelerating

    q_v      = -0;  % reward high absolute velocities
    q_st     =  0;  % penalization of steering
    q_br     =  0;  % penalization of breaking
    q_psidot =  8;  % penalize high yaw rates
    q_acc    = -0;  % reward for accelerating

    % label inputs and outputs
    I_x        = mu_x(1);  % x position in global coordinates
    I_y        = mu_x(2);  % y position in global coordinates
    psi        = mu_x(3);  % yaw
    V_vx       = mu_x(4);  % x velocity in vehicle coordinates
    V_vy       = mu_x(5);  % x velocity in vehicle coordinates
    psidot     = mu_x(6);
    track_dist = mu_x(7);  % track centerline distance
    delta      = u(1);     % steering angle rad2deg(delta)
    T          = u(2);     % torque gain (1=max.acc, -1=max.braking)
    track_vel  = u(3);     % track centerline velocity
    

    % ---------------------------------------------------------------------
    % cost of contour, lag and orientation error
    % ---------------------------------------------------------------------

    % get lag, contour, offroad and orientation error of the vehicle w.r.t.
    % a point in the trajectory that is 'track_dist' far away from the 
    % origin along the track centerline (traveled distance)
    [lag_error, countour_error, offroad_error, orientation_error] = ...
        track.getVehicleDeviation([I_x;I_y], psi, track_dist, eig_XY_max);
    
    cost_contour     = q_c * countour_error^2;
    cost_lag         = q_l * lag_error^2;
    cost_orientation = q_o * orientation_error^2;
    
    % ---------------------------------------------------------------------
    % cost for being outside track
    % ---------------------------------------------------------------------
    % % apply smooth barrier function (we want: offroad_error < 0). 
    % alpha = 40; % smoothing factor... the smaller the smoother
    % offroad_error = (1+exp(-alpha*(offroad_error+0.05))).^-1;
    gamma = 1000;
    lambda = -0.1;
    offroad_error = 5*(sqrt((4+gamma*(lambda-offroad_error).^2)/gamma) - (lambda-offroad_error));
    cost_outside = q_r * offroad_error^2;
    
    % ---------------------------------------------------------------------
    % reward high velocities
    % ---------------------------------------------------------------------
    cost_vel = q_v * norm([V_vx; V_vy]);
    
    % ---------------------------------------------------------------------
    % penalize high yaw rates
    % ---------------------------------------------------------------------
    cost_psidot = q_psidot * psidot^2;
    
    % ---------------------------------------------------------------------
    % reward high track velocities
    % ---------------------------------------------------------------------
    cost_dist = q_d * track_vel;
    
    % ---------------------------------------------------------------------
    % penalize acceleration, braking and steering
    % ---------------------------------------------------------------------
    cost_inputs = (T>0)*q_acc*T^2 + (T<0)*q_br*T^2 + q_st*(delta)^2 ;
    
    % ---------------------------------------------------------------------
    % Calculate final cost
    % ---------------------------------------------------------------------
    cost = cost_contour + ...
           cost_lag + ...
           cost_orientation + ...
           cost_dist + ...
           cost_outside + ...
           cost_inputs + ...
           cost_vel + ...
           cost_psidot;
end