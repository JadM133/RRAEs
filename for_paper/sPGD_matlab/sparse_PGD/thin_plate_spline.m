function  [rbf_eval]= thin_plate_spline(e,r)
    rbf_eval = zeros(size(r));
    e = e(1);
    nz = r~=0; % to deal with singularity
    rbf_eval(nz) = (e.*r(nz)).^2.*log(e.*r(nz));
end % end function thin_plate_spline()