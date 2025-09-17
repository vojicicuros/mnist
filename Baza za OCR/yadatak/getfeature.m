
function[array64]=getfeature(input)    % input 240x240 and output=64
[row,col]=size(input);
basement1=0;
basement2=0;
leaveone=mod(row,8);
leavetwo=mod(col,8);
 for i=1:8
      rowspace=floor(row/8);
      if(leavetwo>0)
          rowspace=rowspace+1;
          leaveone=leaveone-1;
      end
      basement1=basement1+rowspace;
      basement2=0;
       for j=1:8
          colspace=floor(col/8);
          if(leavetwo>0)
              colspace=colspace+1;
              leavetwo=leavetwo-1;
          end
          basement2=basement2+colspace;
          count=0;
          sum=0;
          for k=basement1-rowspace+1 :basement1
              k;
              for l=basement2-colspace+1:basement2
                  l;
                  count=count+input(k,l);
                  sum=sum+1;
              end
          end
          array(i,j)=count/sum;
      end
 end
 array;
  index=1;
  for(i=1:8)
      for(j=1:8)
          array64(index,1)=array(i,j);
          index=index+1;
      end
  end  
  array64;
end
  