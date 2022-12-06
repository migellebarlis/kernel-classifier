n = 60;

im = im2double(imread('GroundTruth1_1_1.jpg'));
ker = im2double(imread('Kernel1G_p.png'));

sz = size(im);

if length(sz) == 3 & sz(3) == 3
    im = rgb2gray(im);
end

x = mean(im,'all');
sd = std(im,1,'all');

im = (im - x) / sd;

figure;
imagesc(fftshift(abs(fft2(im))));

figure;
imagesc(fftshift(abs(fft2(ker))));

figure;
imagesc(fftshift(abs(fft2(conv2(im,ker,'same')))));

% Compute the variance of each patch
var = (conv2(im .^ 2, ones(n), 'valid') - (conv2(im, ones(n), 'valid') .^2 ) / n ) / n;

% Locate the greatest variance
[mx, ind] = max(var, [], 'all');
[r, c] = ind2sub(size(var), ind);

% Take the patch based on the location
pat = im(r:r + n - 1, c:c + n - 1);

% Take the fft of the patch
fft = fftshift(abs(fft2(pat)));

% Show fft
imagesc(fft);