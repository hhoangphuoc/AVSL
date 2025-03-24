echo "Total entries:" && 
wc -l dsfl_laugh/disfluency_laughter_markers.csv && 
echo "Laugh entries:" && 
grep -c ,,1 dsfl_laugh/disfluency_laughter_markers.csv && 
echo "Disfluency entries:" && 
grep -v ,,1 dsfl_laugh/disfluency_laughter_markers.csv | 
grep -v disfluency_type | 
wc -l