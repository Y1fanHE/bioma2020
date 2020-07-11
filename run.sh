# Compare three self-adaptive strategies on Cuckoo Search
python TestCS.py ./SOPyml/SOP1.yml ./ALGOyml/JACS.yml &
python TestCS.py ./SOPyml/SOP2.yml ./ALGOyml/JACS.yml &
python TestCS.py ./SOPyml/SOP3.yml ./ALGOyml/JACS.yml &
python TestCS.py ./SOPyml/SOP4.yml ./ALGOyml/JACS.yml &
python TestCS.py ./SOPyml/SOP1.yml ./ALGOyml/JACS.W1.yml &
python TestCS.py ./SOPyml/SOP2.yml ./ALGOyml/JACS.W1.yml &
python TestCS.py ./SOPyml/SOP3.yml ./ALGOyml/JACS.W1.yml &
python TestCS.py ./SOPyml/SOP4.yml ./ALGOyml/JACS.W1.yml &
wait
python TestCS.py ./SOPyml/SOP1.yml ./ALGOyml/JACS.W2.yml &
python TestCS.py ./SOPyml/SOP2.yml ./ALGOyml/JACS.W2.yml &
python TestCS.py ./SOPyml/SOP3.yml ./ALGOyml/JACS.W2.yml &
python TestCS.py ./SOPyml/SOP4.yml ./ALGOyml/JACS.W2.yml &
python TestCS.py ./SOPyml/SOP1.yml ./ALGOyml/RPCS.yml &
python TestCS.py ./SOPyml/SOP2.yml ./ALGOyml/RPCS.yml &
python TestCS.py ./SOPyml/SOP3.yml ./ALGOyml/RPCS.yml &
python TestCS.py ./SOPyml/SOP4.yml ./ALGOyml/RPCS.yml &
wait
python TestCS.py ./SOPyml/SOP1.yml ./ALGOyml/PECS.yml &
python TestCS.py ./SOPyml/SOP2.yml ./ALGOyml/PECS.yml &
python TestCS.py ./SOPyml/SOP3.yml ./ALGOyml/PECS.yml &
python TestCS.py ./SOPyml/SOP4.yml ./ALGOyml/PECS.yml &
python RunCS1.py &
python RunCS2.py &
python RunCS3.py &
python RunCS4.py &
wait
python RunCS5.py &
python RunCS6.py &
python RunCS7.py &
python RunCS8.py &
wait
cp ./ALGOyml/*JACS* ./tmp/
cp ./ALGOyml/RPCS* ./tmp/
cp ./SOPyml/SOP* ./tmp/
mv tmp sacs
tar -cvzf sacs.tgz sacs/*
rm -rf sacs
echo "Finish SACS testing!"

# Compare three self-adaptive strategies on Differential Evolution
python TestDE.py ./SOPyml/SOP1.yml ./ALGOyml/JADE.yml &
python TestDE.py ./SOPyml/SOP2.yml ./ALGOyml/JADE.yml &
python TestDE.py ./SOPyml/SOP3.yml ./ALGOyml/JADE.yml &
python TestDE.py ./SOPyml/SOP4.yml ./ALGOyml/JADE.yml &
python TestDE.py ./SOPyml/SOP1.yml ./ALGOyml/RPDE.yml &
python TestDE.py ./SOPyml/SOP2.yml ./ALGOyml/RPDE.yml &
python TestDE.py ./SOPyml/SOP3.yml ./ALGOyml/RPDE.yml &
python TestDE.py ./SOPyml/SOP4.yml ./ALGOyml/RPDE.yml &
wait
python TestDE.py ./SOPyml/SOP1.yml ./ALGOyml/PEDE.yml &
python TestDE.py ./SOPyml/SOP2.yml ./ALGOyml/PEDE.yml &
python TestDE.py ./SOPyml/SOP3.yml ./ALGOyml/PEDE.yml &
python TestDE.py ./SOPyml/SOP4.yml ./ALGOyml/PEDE.yml &
wait
cp ./ALGOyml/*DE* ./tmp/
cp ./SOPyml/SOP* ./tmp/
mv tmp sade
tar -cvzf sade.tgz sade/*
rm -rf sade
echo "Finish SADE testing!"

# Compare n_step settings on Parameter Evolved Cuckoo Search
python TestCS.py ./SOPyml/SOP1.yml ./TUNyml/PECS_nstep=1.yml &
python TestCS.py ./SOPyml/SOP2.yml ./TUNyml/PECS_nstep=1.yml &
python TestCS.py ./SOPyml/SOP3.yml ./TUNyml/PECS_nstep=1.yml &
python TestCS.py ./SOPyml/SOP4.yml ./TUNyml/PECS_nstep=1.yml &
python TestCS.py ./SOPyml/SOP1.yml ./TUNyml/PECS_nstep=2.yml &
python TestCS.py ./SOPyml/SOP2.yml ./TUNyml/PECS_nstep=2.yml &
python TestCS.py ./SOPyml/SOP3.yml ./TUNyml/PECS_nstep=2.yml &
python TestCS.py ./SOPyml/SOP4.yml ./TUNyml/PECS_nstep=2.yml &
python TestCS.py ./SOPyml/SOP1.yml ./TUNyml/PECS_nstep=3.yml &
python TestCS.py ./SOPyml/SOP2.yml ./TUNyml/PECS_nstep=3.yml &
wait
python TestCS.py ./SOPyml/SOP3.yml ./TUNyml/PECS_nstep=3.yml &
python TestCS.py ./SOPyml/SOP4.yml ./TUNyml/PECS_nstep=3.yml &
python TestCS.py ./SOPyml/SOP1.yml ./TUNyml/PECS_nstep=4.yml &
python TestCS.py ./SOPyml/SOP2.yml ./TUNyml/PECS_nstep=4.yml &
python TestCS.py ./SOPyml/SOP3.yml ./TUNyml/PECS_nstep=4.yml &
python TestCS.py ./SOPyml/SOP4.yml ./TUNyml/PECS_nstep=4.yml &
python TestCS.py ./SOPyml/SOP1.yml ./TUNyml/PECS_nstep=5.yml &
python TestCS.py ./SOPyml/SOP2.yml ./TUNyml/PECS_nstep=5.yml &
python TestCS.py ./SOPyml/SOP3.yml ./TUNyml/PECS_nstep=5.yml &
python TestCS.py ./SOPyml/SOP4.yml ./TUNyml/PECS_nstep=5.yml &
wait
python TestCS.py ./SOPyml/SOP1.yml ./TUNyml/PECS_nstep=10.yml &
python TestCS.py ./SOPyml/SOP2.yml ./TUNyml/PECS_nstep=10.yml &
python TestCS.py ./SOPyml/SOP3.yml ./TUNyml/PECS_nstep=10.yml &
python TestCS.py ./SOPyml/SOP4.yml ./TUNyml/PECS_nstep=10.yml &
python TestCS.py ./SOPyml/SOP1.yml ./TUNyml/PECS_nstep=20.yml &
python TestCS.py ./SOPyml/SOP2.yml ./TUNyml/PECS_nstep=20.yml &
python TestCS.py ./SOPyml/SOP3.yml ./TUNyml/PECS_nstep=20.yml &
python TestCS.py ./SOPyml/SOP4.yml ./TUNyml/PECS_nstep=20.yml &
python TestCS.py ./SOPyml/SOP1.yml ./TUNyml/PECS_nstep=30.yml &
python TestCS.py ./SOPyml/SOP2.yml ./TUNyml/PECS_nstep=30.yml &
wait
python TestCS.py ./SOPyml/SOP3.yml ./TUNyml/PECS_nstep=30.yml &
python TestCS.py ./SOPyml/SOP4.yml ./TUNyml/PECS_nstep=30.yml &
python TestCS.py ./SOPyml/SOP1.yml ./TUNyml/PECS_nstep=40.yml &
python TestCS.py ./SOPyml/SOP2.yml ./TUNyml/PECS_nstep=40.yml &
python TestCS.py ./SOPyml/SOP3.yml ./TUNyml/PECS_nstep=40.yml &
python TestCS.py ./SOPyml/SOP4.yml ./TUNyml/PECS_nstep=40.yml &
python TestCS.py ./SOPyml/SOP1.yml ./TUNyml/PECS_nstep=50.yml &
python TestCS.py ./SOPyml/SOP2.yml ./TUNyml/PECS_nstep=50.yml &
python TestCS.py ./SOPyml/SOP3.yml ./TUNyml/PECS_nstep=50.yml &
python TestCS.py ./SOPyml/SOP4.yml ./TUNyml/PECS_nstep=50.yml &
wait
mv tmp pecs
tar -czvf PECS_nstep1-50.tgz pecs/*
rm -rf pecs

# Compare n_step settings on Parameter Evolved Differential Evolution
python TestDE.py ./SOPyml/SOP1.yml ./TUNyml/PEDE_nstep=1.yml &
python TestDE.py ./SOPyml/SOP2.yml ./TUNyml/PEDE_nstep=1.yml &
python TestDE.py ./SOPyml/SOP3.yml ./TUNyml/PEDE_nstep=1.yml &
python TestDE.py ./SOPyml/SOP4.yml ./TUNyml/PEDE_nstep=1.yml &
python TestDE.py ./SOPyml/SOP1.yml ./TUNyml/PEDE_nstep=2.yml &
python TestDE.py ./SOPyml/SOP2.yml ./TUNyml/PEDE_nstep=2.yml &
python TestDE.py ./SOPyml/SOP3.yml ./TUNyml/PEDE_nstep=2.yml &
python TestDE.py ./SOPyml/SOP4.yml ./TUNyml/PEDE_nstep=2.yml &
python TestDE.py ./SOPyml/SOP1.yml ./TUNyml/PEDE_nstep=3.yml &
python TestDE.py ./SOPyml/SOP2.yml ./TUNyml/PEDE_nstep=3.yml &
wait
python TestDE.py ./SOPyml/SOP3.yml ./TUNyml/PEDE_nstep=3.yml &
python TestDE.py ./SOPyml/SOP4.yml ./TUNyml/PEDE_nstep=3.yml &
python TestDE.py ./SOPyml/SOP1.yml ./TUNyml/PEDE_nstep=4.yml &
python TestDE.py ./SOPyml/SOP2.yml ./TUNyml/PEDE_nstep=4.yml &
python TestDE.py ./SOPyml/SOP3.yml ./TUNyml/PEDE_nstep=4.yml &
python TestDE.py ./SOPyml/SOP4.yml ./TUNyml/PEDE_nstep=4.yml &
python TestDE.py ./SOPyml/SOP1.yml ./TUNyml/PEDE_nstep=5.yml &
python TestDE.py ./SOPyml/SOP2.yml ./TUNyml/PEDE_nstep=5.yml &
python TestDE.py ./SOPyml/SOP3.yml ./TUNyml/PEDE_nstep=5.yml &
python TestDE.py ./SOPyml/SOP4.yml ./TUNyml/PEDE_nstep=5.yml &
wait
python TestDE.py ./SOPyml/SOP1.yml ./TUNyml/PEDE_nstep=10.yml &
python TestDE.py ./SOPyml/SOP2.yml ./TUNyml/PEDE_nstep=10.yml &
python TestDE.py ./SOPyml/SOP3.yml ./TUNyml/PEDE_nstep=10.yml &
python TestDE.py ./SOPyml/SOP4.yml ./TUNyml/PEDE_nstep=10.yml &
python TestDE.py ./SOPyml/SOP1.yml ./TUNyml/PEDE_nstep=20.yml &
python TestDE.py ./SOPyml/SOP2.yml ./TUNyml/PEDE_nstep=20.yml &
python TestDE.py ./SOPyml/SOP3.yml ./TUNyml/PEDE_nstep=20.yml &
python TestDE.py ./SOPyml/SOP4.yml ./TUNyml/PEDE_nstep=20.yml &
python TestDE.py ./SOPyml/SOP1.yml ./TUNyml/PEDE_nstep=30.yml &
python TestDE.py ./SOPyml/SOP2.yml ./TUNyml/PEDE_nstep=30.yml &
wait
python TestDE.py ./SOPyml/SOP3.yml ./TUNyml/PEDE_nstep=30.yml &
python TestDE.py ./SOPyml/SOP4.yml ./TUNyml/PEDE_nstep=30.yml &
python TestDE.py ./SOPyml/SOP1.yml ./TUNyml/PEDE_nstep=40.yml &
python TestDE.py ./SOPyml/SOP2.yml ./TUNyml/PEDE_nstep=40.yml &
python TestDE.py ./SOPyml/SOP3.yml ./TUNyml/PEDE_nstep=40.yml &
python TestDE.py ./SOPyml/SOP4.yml ./TUNyml/PEDE_nstep=40.yml &
python TestDE.py ./SOPyml/SOP1.yml ./TUNyml/PEDE_nstep=50.yml &
python TestDE.py ./SOPyml/SOP2.yml ./TUNyml/PEDE_nstep=50.yml &
python TestDE.py ./SOPyml/SOP3.yml ./TUNyml/PEDE_nstep=50.yml &
python TestDE.py ./SOPyml/SOP4.yml ./TUNyml/PEDE_nstep=50.yml &
wait
mv tmp pede
tar -cvzf PEDE_nstep1-50.tgz pede/*
rm -rf pede