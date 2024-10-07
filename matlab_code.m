%%%%%% Project assignement 2 %%%%%%
% By Lucie Della Negra
% lude5925@student.su.se
% Due : September 29th
% Last modification : September 22nd

%% Fisher's LDA via a toy problem

rng(0);
% generate data points
muA = [-8;5]; muB = [-3;8];
theta = pi/4;
T = [cos(theta),-sin(theta); sin(theta), cos(theta)];
S = T*diag([1,2])*T';
DataA = S*randn(2, 40) + muA * ones(1,40);
DataB = S*randn(2, 60) + muB * ones(1,60);
% training and testing data: each column is a sample
TrainA = DataA(:,1:end-20);
TestA = DataA(:,end-20+1:end);
TrainB = DataB(:,1:end-20);
TestB = DataB(:,end-20+1:end);
% show data points
figure(1)
hold on;
plot(TrainA(1,:),TrainA(2,:), '+r', 'DisplayName','ClassA');
plot(TrainB(1,:),TrainB(2,:), 'xb', 'DisplayName','ClassB');
axis equal; legend show; title('training data')


%% Task 1: Show that an optimal solution is given by

% sample mean & covariance
mA = mean(TrainA')';
mB = mean(TrainB')';
sA = cov(TrainA');
sB = cov(TrainB');

% separation vector
v = (sA+sB)\(mA-mB);
v = v/norm(v);

%%%%% Task 1
% According to the Fisher's linear discriminant problem, we are looking for
% the best separation vector v for the classes A and B. However, the
% separation ratio can be written as a generalized Rayleigh quotient with :
% S=(mA-mB)(mA-mB).'  and M=(sA+sB)
% By writing the problem this way, the vector v is also an eigenvector
% associated with the largest eigenvalue of inv(M)S so we have:
% inv(M)*S*v=lambda_1*v
% In our case : inv(M)*(mA-mB)*(mA-mB).'*v =lambda*v , as the product 
% (mA-mB).'*v is a scalar, v has to be proportional to inv(M)*(mA-mB).


% show data
projA = v*v'*TrainA; projB = v*v'*TrainB;
projmA = v*v'*mA; projmB = v*v'*mB;


c = (mA+mB)/2;
proj_of_c = v*v.'*c;

figure(2)
hold on;
plot([TrainA(1,:),projA(1,:)], [TrainA(2,:),projA(2,:)], '+r');
plot([TrainB(1,:),projB(1,:)], [TrainB(2,:),projB(2,:)], 'xb');
plot([mA(1),projmA(1)],[mA(2),projmA(2)],'pr','MarkerFaceColor','red','Markersize',10);
plot([mB(1),projmB(1)],[mB(2),projmB(2)],'pb','MarkerFaceColor','blue','Markersize',10);

plot([c(1),proj_of_c(1)],[c(2),proj_of_c(2)],'pg','MarkerFaceColor','green','Markersize',10);
plot([c(1),proj_of_c(1)], [c(2),proj_of_c(2)], '-green');

plot(7*[-v(1),v(1)], 7*[-v(2),v(2)], '-r');

axis equal; title('separation direction and projection')

% By being projected on v, data only have a parameter remaining : the
% position on the line directed by v. As this projection is theorically the
% best projection to separate the data, we can suppose that there exists a
% c such that the proposition is verified. However the data can be to
% sparse or mixed for having a great separation by projection (red cross 
% classified as blue).


%% Task 2 : Build a classifier

function labels = classify(data, v, c_proj)
    data_projection = v.'*data;
    labels = data_projection < c_proj;
end

% Test the classification
classify(TrainB,v,  v.'*c);

% To simplify we consider that a label 0 (resp. 1) is equivalent to class
% A(resp.B)
% Let's compute the success rate : 


function succes_rate = report_success_rate(test_set_0, test_set_1, v, c)
    labels_0 = classify(test_set_0,v, c);
    labels_1 = classify(test_set_1,v, c);

    miss_0 = sum(labels_0(:) == 1);
    miss_1 = sum(labels_1(:) == 0);

    succes_rate = 1 - (miss_0 + miss_1)/(size(test_set_0,2) + size(test_set_1,2));

end

% Compute sucess rate
report_success_rate(TestA, TestB, v, v.'*c)
% Results : 1

%% UCI Benchmark problem
clear
load sonar.mat;
whos('-file','sonar.mat');

load ionosphere.mat;
whos('-file','ionosphere.mat');

%%%%% Task 3
%% On the sonar data

% Start by sort data by labels (to keep balanced datasets)
sonar_data_0 = sonar_data(find(sonar_label(:,1)==0),:);
sonar_data_1 = sonar_data(find(sonar_label(:,1)==1),:);

% Create training datasets
train_sonar_0 = sonar_data_0(1:round(0.7*size(sonar_data_0,1)), :);
train_sonar_1 = sonar_data_1(1:round(0.7*size(sonar_data_1,1)), :);
% Create test datasets
test_sonar_0 = sonar_data_0(round(0.7*size(sonar_data_0,1)):size(sonar_data_0,1), :);
test_sonar_1 = sonar_data_1(round(0.7*size(sonar_data_1,1)):size(sonar_data_1,1), :);

% Sample mean & covariance
m_sonar_0 = mean(train_sonar_0).';
m_sonar_1 = mean(train_sonar_1).';
s_sonar_0 = cov(train_sonar_0);
s_sonar_1 = cov(train_sonar_1);

% Separation vector v
v_sonar = (s_sonar_0+s_sonar_1)\(m_sonar_0-m_sonar_1);
v_sonar = v_sonar/norm(v_sonar);

% Center c
c_sonar = (m_sonar_0+m_sonar_1)/2;

% Compute sucess rate
report_success_rate(test_sonar_0.', test_sonar_1.', v_sonar, v_sonar.'*c_sonar)
% Results : 0.7812

%% Same for the ionosphere

% Start by sort data by labels (to keep balanced datasets)
iono_data_0 = ionosphere_data(find(ionosphere_label(:,1)==0),:);
iono_data_1 = ionosphere_data(find(ionosphere_label(:,1)==1),:);

% Create training datasets
train_iono_0 = iono_data_0(1:round(0.7*size(iono_data_0,1)), :);
train_iono_1 = iono_data_1(1:round(0.7*size(iono_data_1,1)), :);
% Create test datasets
test_iono_0 = iono_data_0(round(0.7*size(iono_data_0,1)):size(iono_data_0,1), :);
test_iono_1 = iono_data_1(round(0.7*size(iono_data_1,1)):size(iono_data_1,1), :);

% Sample mean & covariance
m_iono_0 = mean(train_iono_0).';
m_iono_1 = mean(train_iono_1).';
s_iono_0 = cov(train_iono_0);
s_iono_1 = cov(train_iono_1);

% Separation vector v
v_iono = lsqr(s_iono_0+s_iono_1, m_iono_0-m_iono_1);
% Task 5 : For ionosphere data, Sa+Sb become singular due to the high
% number of zero coefficient. The use of least square method is recommended
% for sparse matrix like Sa+Sb in our case. With the same method as in
% sonar study we obtain a warning as we try to invert a singular matrix.
v_iono = v_iono/norm(v_iono);

% Center c
c_iono = (m_iono_0+m_iono_1)/2;

% Compute sucess rate
report_success_rate(test_iono_0.', test_iono_1.', v_iono, v_iono.'*c_iono)
% Results : 0.8692

%% Task 4 : Try different c values

% For sonar data
c_sonar_values = linspace(v_sonar.'*m_sonar_0, v_sonar.'*m_sonar_1, 1000);
sonar_success_rates=[];

for c_sonar = c_sonar_values
    sonar_success_rates = [sonar_success_rates report_success_rate(test_sonar_0.', test_sonar_1.', v_sonar, c_sonar)];
end

% For iono data
c_iono_values = linspace(v_iono.'*m_iono_0, v_iono.'*m_iono_1, 1000);
iono_success_rates=[];

for c_iono = c_iono_values
    iono_success_rates = [iono_success_rates report_success_rate(test_iono_0.', test_iono_1.', v_iono, c_iono)];
end

% Plots
figure(3)
plot(c_sonar_values, sonar_success_rates)
title('Success rate for different c values (sonar data)')
ylabel('Success rate')
xlabel('Different c values')
figure(4)
plot(c_iono_values, iono_success_rates)
title('Success rate for different c values (ionosphere data)')
ylabel('Success rate')
xlabel('Different c values')


