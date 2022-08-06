function out = stretch_img(u)

% out = stretch_img(u) produces an image for visualizing a matrix u in 
% stretching the value of u so that its min(u) and max(u) become 0 and 255,
% and then apply the uint8 conversion.
% If u is constant, then out is zero.

if(~isreal(u))
    error('[stretch_img] The image u should only have real value');
end

mi = min(u(:));
ma = max(u(:));
if(ma-mi>0)
    out = 255/(ma-mi)*(u-mi);
else
    out = zeros(size(u));
end
out = uint8(out);
end