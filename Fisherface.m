%%
% Read Images
clc
clear
[im, person, number, subset] = readFaceImages('faces');

%%
% Vectorize images
im_subset_1 = [];
im_subset_2 = [];
im_subset_3 = [];
im_subset_4 = [];
im_subset_5 = [];
for i = 1:size(im,2)
    if subset(i) == 1
           im_subset_1 = [im_subset_1;person(i),reshape(cell2mat(im(i)),[1,2500])];
    elseif subset(i) == 2
           im_subset_2 = [im_subset_2;person(i),reshape(cell2mat(im(i)),[1,2500])];
    elseif subset(i) == 3
           im_subset_3 = [im_subset_3;person(i),reshape(cell2mat(im(i)),[1,2500])];
    elseif subset(i) == 4
           im_subset_4 = [im_subset_4;person(i),reshape(cell2mat(im(i)),[1,2500])];
    elseif subset(i) == 5
           im_subset_5 = [im_subset_5;person(i),reshape(cell2mat(im(i)),[1,2500])];
    end
end

%%
% Subtracting mean from training images
X = [];
clear mean
im_mean = mean(im_subset_1(:,2:2501));
im_std = std(im_subset_1(:,2:2501));
for i = 1:size(im_subset_1,1)
    X = [X; mat2gray((im_subset_1(i,2:2501) - im_mean)./im_std)];
end
%{ 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
UNCOMMENT TO TRAIN WITH BOTH SUBSET 1 AND 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(im_subset_5,1)
    X = [X; mat2gray((im_subset_5(i,2:2501) - im_mean)./im_std)];
end

im_mean_face = reshape(im_mean,[50,50]);
figure();
imshow(im_mean_face);
%}
%%
% Compute EigenVectors
[U,L] = eig(X'*X);
V = X*U;
j = 1;
c = 31;
X_efaces = [];
for i = 1:c
    X_pca = V(:,2496-i)'*X;
    X_efaces = [X_efaces; X_pca];
    X_reshape = reshape(X_pca,[50,50]);
    X_eface = mat2gray(X_reshape);
    %subplot(2,5,j), imshow(X_eface);
    j = j+1;
end

%%
% Calculating Optimum Matrix W_opt
% Training SET
W_pca = [];
for i = 1:size(im_subset_1,1)
    im_red = im_subset_1(i,1);
    for j = 1:size(X_efaces,1)
        im_red = [im_red,dot(X_efaces(j,:)',( im_subset_1(i,2:2501)- im_mean)./im_std)];
    end
    W_pca = [W_pca;im_red];
end
%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
UNCOMMENT TO TRAIN WITH BOTH SUBSET 1 AND 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(im_subset_5,1)
    im_red = im_subset_5(i,1);
    for j = 1:size(X_efaces,1)
        im_red = [im_red,dot(X_efaces(j,:)',( im_subset_5(i,2:2501)- im_mean)./im_std)];
    end
    W_pca = [W_pca;im_red];
end
%}
% Calculating Scatter Matrices
avg_set = mean(W_pca(:,2:c+1));
avg_class = [];
class_mat = [];
count = 0;
class_sample_count = [];
for i = 1:size(W_pca,1)-1
    class_mat = [class_mat;W_pca(i,:)];
    count = count+1;
    if W_pca(i+1,1) ~= W_pca(i,1)
        avg_class = [avg_class; class_mat(1,1), mean(class_mat(:,2:c+1))];
        class_mat = [];
        class_sample_count = [class_sample_count;count];
        count = 0;
    elseif i == size(W_pca,1)-1
        class_mat = [class_mat;W_pca(i+1,:)];
        avg_class = [avg_class; class_mat(1,1), mean(class_mat(:,2:c+1))];
        count = count+1;
        class_sample_count = [class_sample_count;count];
    end
end

Scatter_i = [];
scatter_matrix = 0;
for i = 1:size(W_pca,1)-1
    x_minus_u = W_pca(i,2:c+1)-avg_class(W_pca(i,1),2:c+1);
    scatter_matrix = scatter_matrix + (x_minus_u'*x_minus_u);
    if W_pca(i+1,1) ~= W_pca(i,1)
        Scatter_i = [Scatter_i;scatter_matrix];
        scatter_matrix = 0;
    elseif i == size(W_pca,1)-1
        x_minus_u = W_pca(i+1,2:c+1)-avg_class(W_pca(i+1,1),2:c+1);
        scatter_matrix = scatter_matrix + (x_minus_u*x_minus_u');
        Scatter_i = [Scatter_i;scatter_matrix];
    end
end

Sw = 0;
for i = 1:c:size(Scatter_i,1)
    Sw = Sw + Scatter_i(i:i+(c-1),:);
end

Sb = 0;
for i = 1:size(avg_class,1)
    N_i = class_sample_count(i,1);
    Sb = Sb + N_i.*((avg_class(i,2:c+1)-avg_set)'*(avg_class(i,2:c+1)-avg_set));
end

[W_fld, Df] = eig(Sb,Sw);
W_pca = W_pca(:,2:c+1);
W_opt = W_fld'*X_efaces;

%%
% Reduce dimensionality

% All sets
c_dim_subset1 = [];
c_dim_subset2 = [];
c_dim_subset3 = [];
c_dim_subset4 = [];
c_dim_subset5 = [];

for i = 1:size(im_subset_1,1)
    im_red = im_subset_1(i,1);
    for j = 1:size(W_opt,1)
        im_red = [im_red,dot(W_opt(j,:),(( im_subset_1(i,2:2501)- im_mean)./im_std)')];
    end
    c_dim_subset1 = [c_dim_subset1;im_red];
end

for i = 1:size(im_subset_2,1)
    im_red = im_subset_2(i,1);
    for j = 1:size(W_opt,1)
        im_red = [im_red,dot(W_opt(j,:),(( im_subset_2(i,2:2501)- im_mean)./im_std)')];
    end
    c_dim_subset2 = [c_dim_subset2;im_red];
end
for i = 1:size(im_subset_3,1)
    im_red = im_subset_3(i,1);
    for j = 1:size(W_opt,1)
        im_red = [im_red,dot(W_opt(j,:),(( im_subset_3(i,2:2501)- im_mean)./im_std)')];
    end
    c_dim_subset3 = [c_dim_subset3;im_red];
end
for i = 1:size(im_subset_4,1)
    im_red = im_subset_4(i,1);
    for j = 1:size(W_opt,1)
        im_red = [im_red,dot(W_opt(j,:),(( im_subset_4(i,2:2501)- im_mean)./im_std)')];
    end
    c_dim_subset4 = [c_dim_subset4;im_red];
end
for i = 1:size(im_subset_5,1)
    im_red = im_subset_5(i,1);
    for j = 1:size(W_opt,1)
        im_red = [im_red,dot(W_opt(j,:),(( im_subset_5(i,2:2501)- im_mean)./im_std)')];
    end
    c_dim_subset5 = [c_dim_subset5;im_red];
end

%%
%  Evaluation

c_dim_subset1_match = [];
c_dim_subset2_match = [];
c_dim_subset3_match = [];
c_dim_subset4_match = [];
c_dim_subset5_match = [];
match_accuracy = [];

% Evaluating subset 1
for i = 1:size(c_dim_subset1,1)
    dist_n = [];
    for j = 1:size(c_dim_subset1,1)
        dist_n = [dist_n,norm(c_dim_subset1(i,2:c+1)-c_dim_subset1(j,2:c+1))];
    end
    [min_dist, ind] = min(dist_n);
    c_dim_subset1_match = [c_dim_subset1_match;c_dim_subset1(i,1),c_dim_subset1(ind,1)]; 
end
count = 0;
for i = 1:size(c_dim_subset1_match,1)
    if c_dim_subset1_match(i,1) == c_dim_subset1_match(i,2)
       count = count+1;
    end
end
match_accuracy = [match_accuracy, count/size(c_dim_subset1_match,1)];

% Evaluating subset 2
for i = 1:size(c_dim_subset2,1)
    dist_n = [];
    for j = 1:size(c_dim_subset1,1)
        dist_n = [dist_n,norm(c_dim_subset2(i,2:c+1)-c_dim_subset1(j,2:c+1))];
    end
    [min_dist, ind] = min(dist_n);
    c_dim_subset2_match = [c_dim_subset2_match;c_dim_subset2(i,1),c_dim_subset1(ind,1)]; 
end
count = 0;
for i = 1:size(c_dim_subset2_match,1)
    if c_dim_subset2_match(i,1) == c_dim_subset2_match(i,2)
       count = count+1;
    end
end
match_accuracy = [match_accuracy, count/size(c_dim_subset2_match,1)];

% Evaluating subset 3
for i = 1:size(c_dim_subset3,1)
    dist_n = [];
    for j = 1:size(c_dim_subset1,1)
        dist_n = [dist_n,norm(c_dim_subset3(i,2:c+1)-c_dim_subset1(j,2:c+1))];
    end
    [min_dist, ind] = min(dist_n);
    c_dim_subset3_match = [c_dim_subset3_match;c_dim_subset3(i,1),c_dim_subset1(ind,1)]; 
end
count = 0;
for i = 1:size(c_dim_subset3_match,1)
    if c_dim_subset3_match(i,1) == c_dim_subset3_match(i,2)
       count = count+1;
    end
end
match_accuracy = [match_accuracy, count/size(c_dim_subset3_match,1)];

% Evaluating subset 4
for i = 1:size(c_dim_subset4,1)
    dist_n = [];
    for j = 1:size(c_dim_subset1,1)
        dist_n = [dist_n,norm(c_dim_subset4(i,2:c+1)-c_dim_subset1(j,2:c+1))];
    end
    [min_dist, ind] = min(dist_n);
    c_dim_subset4_match = [c_dim_subset4_match;c_dim_subset4(i,1),c_dim_subset1(ind,1)]; 
end
count = 0;
for i = 1:size(c_dim_subset4_match,1)
    if c_dim_subset4_match(i,1) == c_dim_subset4_match(i,2)
       count = count+1;
    end
end
match_accuracy = [match_accuracy, count/size(c_dim_subset4_match,1)];

% Evaluating subset 5
for i = 1:size(c_dim_subset5,1)
    dist_n = [];
    for j = 1:size(c_dim_subset1,1)
        dist_n = [dist_n,norm(c_dim_subset5(i,2:c+1)-c_dim_subset1(j,2:c+1))];
    end
    [min_dist, ind] = min(dist_n);
    c_dim_subset5_match = [c_dim_subset5_match;c_dim_subset5(i,1),c_dim_subset1(ind,1)]; 
end
count = 0;
for i = 1:size(c_dim_subset5_match,1)
    if c_dim_subset5_match(i,1) == c_dim_subset5_match(i,2)
       count = count+1;
    end
end
match_accuracy = [match_accuracy, count/size(c_dim_subset5_match,1)]