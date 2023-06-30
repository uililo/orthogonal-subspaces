echo "creating utt center"
python standardize_by_utterance.py features/$2/$1/orig features_new/$2/$1/utt_center

echo "creating utt std"
python standardize_by_utterance.py features/$2/$1/orig features_new/$2/$1/utt_std --rescale

echo "compute speaker mean"
python compute_speaker_mean.py features/$2/$1/orig LibriSpeech/forced_alignment/$2.ali 

echo "creating spk center"
python standardize_by_speaker.py features/$2/$1/orig features_new/$2/cpc_big/spk_center

echo "creating spk std"
python standardize_by_speaker.py features/$2/$1/orig features_new/$2/$1/spk_std --rescale

echo "done"