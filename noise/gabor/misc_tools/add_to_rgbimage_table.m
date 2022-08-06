function imtableout = add_to_rgbimage_table(im, imtable, marg, mt, nt)

% Add the rgb image im to the table imtable with margin marg.
% (mt, nt) are the coordinates in the image table.

% The input image im can be grayscale.
% The dimensions are supposed to match.


Mim = size(im,1);
Nim = size(im,2);


mind = (mt-1)*Mim + (mt-1)*marg;
nind = (nt-1)*Nim + (nt-1)*marg;

imtableout = imtable;

if(size(im,3) == 3)
    imtableout((mind+1:mind+Mim), (nind+1:nind+Nim), :) = im;
    return;
elseif(size(im,3) == 1)
    for c = 1:3
        imtableout((mind+1:mind+Mim), (nind+1:nind+Nim), c) = im;
    end
    return;
else
    error('Dimension problem with the image im (neither grayscale nor rgb)');
end

end