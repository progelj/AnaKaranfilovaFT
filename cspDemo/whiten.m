function W2 = whiten( sigma )
%WHITEN whiteneing of data from covariance matrix
% input:
%    sigma - covariance matrix
% output
%    whitening matrix
% W'* sigma * W = I

[U, S ,~] = svd(sigma);
%W = U * S^(-0.5);

%ZCA Whitening:
epsilon = 1e-5;
W2 = U * (S + eye(size(U,1)).*epsilon)^(-0.5) * U';

%figure(100);subplot(1,2,1);imagesc(W); colorbar; subplot(1,2,2); imagesc(W2); colorbar

end
