function [converted, standard] = convertreward(rewardvector)
%converts all reward volumes to reward levels
%CG moved here 11/5/21

converted = rewardvector;

converted(converted==4) = 1;
converted(converted==5) = 1;
converted(converted==8) = 2;
converted(converted==10) = 2;
converted(converted==16) = 3;
converted(converted==20) = 3;
converted(converted==32) = 4;
converted(converted==40) = 4;
converted(converted==64) = 5;
converted(converted==80) = 5;

standard = converted;
standard(converted==1) = 5;
standard(converted==2) = 10;
standard(converted==3) = 20;
standard(converted==4) = 40;
standard(converted==5) = 80;

end