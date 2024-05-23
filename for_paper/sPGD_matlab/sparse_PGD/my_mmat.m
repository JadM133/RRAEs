function [C] = my_mmat(A, B)
C = zeros(size(A, 1), size(B, 2), size(A, 3));
for ii = 1: size(A, 3)
    C(:, :, ii) = A(:, :, ii)*B(:, :, ii);
end
end