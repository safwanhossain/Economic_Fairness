mkdir Outputs

for r in {1..25}
do 
	echo
	echo Run $r
	echo

	mkdir Run$r

	for n in 10 20 30 40 50 60 70 80 90 100 120 150
	do
		echo $n

		mkdir Run$r/Size$n
		python generateData_mixture.py $n 10 10 5 > Run$r/Size$n/generator_output.txt
		python learningMixture.py 10.0 5 load > Run$r/Size$n/learning_output.txt

		mv L.csv Run$r/Size$n/L.csv
		mv u.csv Run$r/Size$n/u.csv
		mv X.csv Run$r/Size$n/X.csv

		mv Outputs/* Run$r/Size$n/
	done

done