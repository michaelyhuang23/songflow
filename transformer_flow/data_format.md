### Square Mel Jamendo Data

Square jamendo data uses preprocess_data.py with
sampling_rate = 44100 
n_fft = 1000
hop_length=n_fft-100
n_mels=500

these are important information for inverting the data. The x (cos) is concatenated with the y (sin)
The idea of this dataset is to make the output (1000, 491) close to a square. In the case where we use a largish hidden_dim, the D is not too large

### Square Jamendo Data

Change preprocess_data.py such that we don't apply the mel filter. This preserves the data quality but might make it harder to interpret
sampling_rate = 44100 
n_fft = 1000
hop_length=n_fft-100