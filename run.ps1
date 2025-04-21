python main.py --root_path "D:\MINHNHAT\data" `
  --video_path "D:\MINHNHAT\archive\out" `
  --annotation_path "D:\MINHNHAT\archive\json\ucf101_01.json" `
  --result_path "D:\MINHNHAT\data\results" `
  --dataset ucf101 `
  --model efficientnet --efficientnet_version 4 `
  --n_classes 101 --n_epochs 100 `
  --batch_size 8 --checkpoint 5 `
  --no_val | Tee-Object -Append "D:\MINHNHAT\data\results\script_output.log"
