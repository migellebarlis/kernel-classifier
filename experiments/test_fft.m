n = 60;

im = im2double(imread('GroundTruth1_1_1.jpg'));
ker = im2double(imread('Kernel1G_p.png'));

sz = size(im);

if length(sz) == 3 && sz(3) == 3
    im = rgb2gray(im);
end

figure
subplot(2,5,1)
imshow(im)
title('image')

subplot(2,5,2)
imshow(ker)
title('kernel')

subplot(2,5,6)
imagesc(fftshift(abs(fft2(im))))
title('image fft')

subplot(2,5,7)
imagesc(fftshift(abs(fft2(ker))))
title('kernel fft')

% Blur the image
im = conv2(im,ker,'same');

subplot(2,5,3)
imshow(im)
title('convolved')

subplot(2,5,8)
imagesc(fftshift(abs(fft2(im))))
title('convolved fft')

% Normalise the blurred image
x = mean(im,'all');
sd = std(im,1,'all');

im = (im - x) / sd;

subplot(2,5,4)
imshow(im)
title('normalised')

subplot(2,5,9)
imagesc(fftshift(abs(fft2(im))))
title('normalised fft')

% Compute the variance of each patch
var = (conv2(im .^ 2, ones(n), 'valid') - (conv2(im, ones(n), 'valid') .^2 ) / n ) / n;

% Locate the greatest variance
[mx, ind] = max(var, [], 'all');
[r, c] = ind2sub(size(var), ind);

% Take the patch based on the location
pat = im(r:r + n - 1, c:c + n - 1);

subplot(2,5,5)
imshow(pat)
title('patch')

% Take the fft of the patch
fft = fftshift(abs(fft2(pat)));

subplot(2,5,10)
imagesc(fft)
title('patch fft')