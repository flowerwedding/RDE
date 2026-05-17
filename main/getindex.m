function [r1, r2, r3]  = getindex(popuSize)

r1 = zeros(1, popuSize);
r2 = zeros(1, popuSize);
r3 = zeros(1, popuSize);

for i = 1 : popuSize
    
    sequence = 1 : popuSize;
    sequence(i) = [];

    temp = floor(rand * (popuSize - 1)) + 1;
    r1(i) = sequence(temp);
    sequence(temp) = [];

    temp = floor(rand * (popuSize - 2)) + 1;
    r2(i) = sequence(temp);
    sequence(temp) = [];

    temp = floor(rand * (popuSize - 3)) + 1;
    r3(i) = sequence(temp);

end