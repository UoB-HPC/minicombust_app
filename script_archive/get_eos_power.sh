FILE=$1

CPU0=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$2;count+=1}END{print sum/count}'`
CPU1=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$3;count+=1}END{print sum/count}'`

GB_SXM1=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$4;count+=1}END{print sum/count}'`
GB_SXM2=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$5;count+=1}END{print sum/count}'`
GB_SXM3=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$6;count+=1}END{print sum/count}'`
GB_SXM4=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$7;count+=1}END{print sum/count}'`
GB_SXM5=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$8;count+=1}END{print sum/count}'`
GB_SXM6=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$9;count+=1}END{print sum/count}'`
GB_SXM7=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$10;count+=1}END{print sum/count}'`
GB_SXM8=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$11;count+=1}END{print sum/count}'`

PSU0=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$12;count+=1}END{print sum/count}'`
PSU1=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$13;count+=1}END{print sum/count}'`
PSU2=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$14;count+=1}END{print sum/count}'`
PSU3=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$15;count+=1}END{print sum/count}'`
PSU4=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$16;count+=1}END{print sum/count}'`
PSU5=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$17;count+=1}END{print sum/count}'`

PDU0=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$18;count+=1}END{print sum/count}'`
PDU1=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$19;count+=1}END{print sum/count}'`
PDU2=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$20;count+=1}END{print sum/count}'`
PDU3=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$21;count+=1}END{print sum/count}'`
PDU4=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$22;count+=1}END{print sum/count}'`
PDU5=`grep -v timestamp $FILE | awk -F, 'BEGIN{sum=0;count=0}{sum+=$23;count+=1}END{print sum/count}'`


CPU=$(bc -l <<<$CPU0+$CPU1)
GPU=$(bc -l <<<$GB_SXM1+$GB_SXM2+$GB_SXM3+$GB_SXM4+$GB_SXM5+$GB_SXM6+$GB_SXM7+$GB_SXM8)
PSU=$(bc -l <<<$PSU0+$PSU1+$PSU2+$PSU3+$PSU4+$PSU5)
PDU=$(bc -l <<<$PDU0+$PDU1+$PDU2+$PDU3+$PDU4+$PDU5)

echo "CPU:" $CPU "W"
echo "GPU:" $GPU "W"
echo "PSU:" $PSU "W"
echo "PDU:" $PDU "W"
