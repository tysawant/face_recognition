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
%}
im_mean_face = reshape(im_mean,[50,50]);
figure();
imshow(im_mean_face);

%%
% Compute EigenFaces
[U,L] = eig(X'*X);
V = X*U;
j = 1;
k = 30;
X_efaces = [];
figure();
for i = 1:k
    X_pca = V(:,2496-i)'*X;
    X_efaces = [X_efaces; X_pca];
    X_reshape = reshape(X_pca,[50,50]);
    X_eface = mat2gray(X_reshape);
    subplot(3,k/3,j), imshow(X_eface);
    j = j+1;
end

%%
% Reduce dimensionality
% Training SET
k_dim_subset1 = [];
for i = 1:size(im_subset_1,1)
    im_red = im_subset_1(i,1);
    for j = 1:size(X_efaces,1)
        im_red = [im_red,dot(X_efaces(j,:)',( im_subset_1(i,2:2501)- im_mean)./im_std)];
    end
    k_dim_subset1 = [k_dim_subset1;im_red];
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
    k_dim_subset1 = [k_dim_subset1;im_red];
end
%}

% All sets
k_dim_subset2 = [];
k_dim_subset3 = [];
k_dim_subset4 = [];
k_dim_subset5 = [];
for i = 1:size(im_subset_2,1)
    im_red = im_subset_2(i,1);
    for j = 1:size(X_efaces,1)
        im_red = [im_red,dot(X_efaces(j,:)',( im_subset_2(i,2:2501)- im_mean)./im_std)];
    end
    k_dim_subset2 = [k_dim_subset2;im_red];
end
for i = 1:size(im_subset_3,1)
    im_red = im_subset_3(i,1);
    for j = 1:size(X_efaces,1)
        im_red = [im_red,dot(X_efaces(j,:)',( im_subset_3(i,2:2501)- im_mean)./im_std)];
    end
    k_dim_subset3 = [k_dim_subset3;im_red];
end
for i = 1:size(im_subset_4,1)
    im_red = im_subset_4(i,1);
    for j = 1:size(X_efaces,1)
        im_red = [im_red,dot(X_efaces(j,:)',( im_subset_4(i,2:2501)- im_mean)./im_std)];
    end
    k_dim_subset4 = [k_dim_subset4;im_red];
end
for i = 1:size(im_subset_5,1)
    im_red = im_subset_5(i,1);
    for j = 1:size(X_efaces,1)
        im_red = [im_red,dot(X_efaces(j,:)',( im_subset_5(i,2:2501)- im_mean)./im_std)];
    end
    k_dim_subset5 = [k_dim_subset5;im_red];
end

%%
%  Evaluation

k_dim_subset1_match = [];
k_dim_subset2_match = [];
k_dim_subset3_match = [];
k_dim_subset4_match = [];
k_dim_subset5_match = [];
match_accuracy = [];

% Evaluating subset 1
for i = 1:size(k_dim_subset1,1)
    dist_n = [];
    for j = 1:size(k_dim_subset1,1)
        dist_n = [dist_n,norm(k_dim_subset1(i,2:k+1)-k_dim_subset1(j,2:k+1))];
    end
    [min_dist, ind] = min(dist_n);
    k_dim_subset1_match = [k_dim_subset1_match;k_dim_subset1(i,1),k_dim_subset1(ind,1)]; 
end
c = 0;
for i = 1:size(k_dim_subset1_match,1)
    if k_dim_subset1_match(i,1) == k_dim_subset1_match(i,2)
       c = c+1;
    end
end
match_accuracy = [match_accuracy, c/size(k_dim_subset1_match,1)];

% Evaluating subset 2
for i = 1:size(k_dim_subset2,1)
    dist_n = [];
    for j = 1:size(k_dim_subset1,1)
        dist_n = [dist_n,norm(k_dim_subset2(i,2:k+1)-k_dim_subset1(j,2:k+1))];
    end
    [min_dist, ind] = min(dist_n);
    k_dim_subset2_match = [k_dim_subset2_match;k_dim_subset2(i,1),k_dim_subset1(ind,1)]; 
end
c = 0;
for i = 1:size(k_dim_subset2_match,1)
    if k_dim_subset2_match(i,1) == k_dim_subset2_match(i,2)
       c = c+1;
    end
end
match_accuracy = [match_accuracy, c/size(k_dim_subset2_match,1)];

% Evaluating subset 3
for i = 1:size(k_dim_subset3,1)
    dist_n = [];
    for j = 1:size(k_dim_subset1,1)
        dist_n = [dist_n,norm(k_dim_subset3(i,2:k+1)-k_dim_subset1(j,2:k+1))];
    end
    [min_dist, ind] = min(dist_n);
    k_dim_subset3_match = [k_dim_subset3_match;k_dim_subset3(i,1),k_dim_subset1(ind,1)]; 
end
c = 0;
for i = 1:size(k_dim_subset3_match,1)
    if k_dim_subset3_match(i,1) == k_dim_subset3_match(i,2)
       c = c+1;
    end
end
match_accuracy = [match_accuracy, c/size(k_dim_subset3_match,1)];

% Evaluating subset 4
for i = 1:size(k_dim_subset4,1)
    dist_n = [];
    for j = 1:size(k_dim_subset1,1)
        dist_n = [dist_n,norm(k_dim_subset4(i,2:k+1)-k_dim_subset1(j,2:k+1))];
    end
    [min_dist, ind] = min(dist_n);
    k_dim_subset4_match = [k_dim_subset4_match;k_dim_subset4(i,1),k_dim_subset1(ind,1)]; 
end
c = 0;
for i = 1:size(k_dim_subset4_match,1)
    if k_dim_subset4_match(i,1) == k_dim_subset4_match(i,2)
       c = c+1;
    end
end
match_accuracy = [match_accuracy, c/size(k_dim_subset4_match,1)];

% Evaluating subset 5
for i = 1:size(k_dim_subset5,1)
    dist_n = [];
    for j = 1:size(k_dim_subset1,1)
        dist_n = [dist_n,norm(k_dim_subset5(i,2:k+1)-k_dim_subset1(j,2:k+1))];
    end
    [min_dist, ind] = min(dist_n);
    k_dim_subset5_match = [k_dim_subset5_match;k_dim_subset5(i,1),k_dim_subset1(ind,1)]; 
end
c = 0;
for i = 1:size(k_dim_subset5_match,1)
    if k_dim_subset5_match(i,1) == k_dim_subset5_match(i,2)
       c = c+1;
    end
end
match_accuracy = [match_accuracy, c/size(k_dim_subset5_match,1)]

%%
% Reconstrucion
im_rec_1 = zeros(1,2500);
im_rec_2 = zeros(1,2500);
im_rec_3 = zeros(1,2500);
im_rec_4 = zeros(1,2500);
im_rec_5 = zeros(1,2500);

for j = 1:k
    im_rec_1 =  im_rec_1 + (k_dim_subset1(1,j+1)*X_efaces(j,:));
    im_rec_2 =  im_rec_2 + (k_dim_subset2(1,j+1)*X_efaces(j,:));
    im_rec_3 =  im_rec_3 + (k_dim_subset3(1,j+1)*X_efaces(j,:));
    im_rec_4 =  im_rec_4 + (k_dim_subset4(1,j+1)*X_efaces(j,:));
    im_rec_5 =  im_rec_5 + (k_dim_subset5(1,j+1)*X_efaces(j,:));
end
    
im_rec_1 = reshape(mat2gray(im_mean + mat2gray(im_rec_1)),[50,50]);
im_rec_2 = reshape(mat2gray(im_mean + mat2gray(im_rec_2)),[50,50]);
im_rec_3 = reshape(mat2gray(im_mean + mat2gray(im_rec_3)),[50,50]);
im_rec_4 = reshape(mat2gray(im_mean + mat2gray(im_rec_4)),[50,50]);
im_rec_5 = reshape(mat2gray(im_mean + mat2gray(im_rec_5)),[50,50]);
figure();

subplot(2,5,1), imshow(reshape(im_subset_1(1,2:2501),[50,50]));
subplot(2,5,2), imshow(reshape(im_subset_2(1,2:2501),[50,50]));
subplot(2,5,3), imshow(reshape(im_subset_3(1,2:2501),[50,50]));
subplot(2,5,4), imshow(reshape(im_subset_4(1,2:2501),[50,50]));
subplot(2,5,5), imshow(reshape(im_subset_5(1,2:2501),[50,50]));
subplot(2,5,6), imshow(im_rec_1);
subplot(2,5,7), imshow(im_rec_2);
subplot(2,5,8), imshow(im_rec_3);
subplot(2,5,9), imshow(im_rec_4);
subplot(2,5,10), imshow(im_rec_5);
