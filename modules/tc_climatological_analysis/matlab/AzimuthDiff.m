function AZdiff = AzimuthDiff(hding,hdgr)

% AzimuthDiff.m
% By: Norberto C. Nadal-Caraballo, PhD

for h = 1:size(hding,2)
    
    for hh = 1:size(hdgr,1)
        
        dgr = hdgr(hh,1);
        
        if (dgr == 0)
            hdiff(hh,h) = abs(dgr-hding(1,h));
            
        elseif (dgr > 0 && dgr <= 90 && hding(1,h) <= (dgr-180))
            hdiff(hh,h) = abs(360-dgr+hding(1,h));  
        elseif (dgr > 0 && dgr <= 90 && hding(1,h) > (dgr-180))
            hdiff(hh,h) = abs(dgr-hding(1,h));  
            
        elseif (dgr < 0 && dgr >= -90 && hding(1,h) <= (dgr+180))
            hdiff(hh,h) = abs(dgr-hding(1,h)); 
        elseif (dgr < 0 && dgr >= -90 && hding(1,h) > (dgr+180))
            hdiff(hh,h) = abs(360+dgr-hding(1,h)); 
                 
        elseif (dgr > 90 && dgr <= 180 && hding(1,h) <= (dgr-180))
            hdiff(hh,h) = abs(360-dgr+hding(1,h));
        elseif (dgr > 90 && dgr <= 180 && hding(1,h) > (dgr-180))
            hdiff(hh,h) = abs(dgr-hding(1,h));             
            
        elseif (dgr < -90 && dgr >= -180 && hding(1,h) <= (dgr+180))
            hdiff(hh,h) = abs(dgr-hding(1,h)); 
        elseif (dgr < -90 && dgr >= -180 && hding(1,h) > (dgr+180))
             hdiff(hh,h) = abs(360+dgr-hding(1,h)); 
                     
            
        else
            hdiff(hh,h) = NaN;
        end%if
        
    end%for hh
    
end%for h

AZdiff = hdiff;

%% END

