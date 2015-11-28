function D = distMat(P1, P2)

% Computing pairwise Euclidian distances between two sets of vectors
% Each vector is one column
  
P1 = double(P1);
P2 = double(P2);

X1 = sum(P1.^2);
X2 = sum(P2.^2);

R=P1'*P2;

D=real(sqrt(bsxfun(@plus, X1', X2)-2*R));
