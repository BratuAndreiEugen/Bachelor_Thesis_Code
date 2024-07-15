insert into Datasets(name, nr_classes, type)
values ('irmas_novoice', 10, 'Instrument'),
('philarmonia', 20, 'Instrument'),
('kaggle_instruments', 10, 'Instrument'),
('gtzan', 10, 'Genre')

insert into Models(name, train_acc, validation_acc, train_dataset_id, input_length_seconds, input_shape, feature_description)
values ('irmas_svm', 96.8, 70.15, 1, 3, '55', 'Best Feature Set'),
('irmas_rf', 99.58, 59.52, 1, 3, '55', 'Best Feature Set'),
('irmas_knn', 99.9, 67.95, 1, 3, '55', 'Best Feature Set'),
('philarmonia_svm', 96.69, 98.35, 2, 3, '55', 'Best Feature Set'),
('philarmonia_rf', 99.92, 95.39, 2, 3, '55', 'Best Feature Set'),
('philarmonia_knn', 99.99, 94.40, 2, 3, '55', 'Best Feature Set'),
('kaggle_svm', 96.62, 80.00, 3, 3, '55', 'Best Feature Set'),
('kaggle_rf', 99.25, 80.00, 3, 3, '55', 'Best Feature Set'),
('kaggle_knn', 100.00, 83.33, 3, 3, '55', 'Best Feature Set'),
('gtzan_svm', 99.41, 91.69, 4, 3, '55', 'Best Feature Set'),
('gtzan_rf', 99.66, 83.38, 4, 3, '55', 'Best Feature Set'),
('gtzan_knn', 99.92, 92.09, 4, 3, '55', 'Best Feature Set'),
('irmas_cnng1', 58.38, 50.08, 1, 3, '130,13,1', '13 MFCC'),
('gtzan_cnng1', 76.15, 71.97, 4, 3, '130,13,1', '13 MFCC'),
('philarmonia_cnng1', 95.90, 96.38, 2, 3, '130,13,1', '13 MFCC'),
('irmas_rnng1', 49.91, 40.89, 1, 3, '130,13', '13 MFCC'),
('gtzan_rnng1', 70.22, 65.67, 4, 3, '130,13', '13 MFCC'),
('philarmonia_rnng1', 59.51, 62.95, 2, 3, '130,13', '13 MFCC'),
('irmas_resnet', 59.61, 52.19, 1, 3, '130,128,3', 'Mel Spectrograms'),
('gtzan_resnet', 59.02, 55.01, 4, 3, '130,128,3', 'Mel Spectrograms'),
('philarmonia_resnet', 96.34, 92.05, 2, 3, '130,128,3', 'Mel Spectrograms'),
('kaggle_cnn_one', 96.62, 80.00, 3, 0.1, '9,13,1', '13 MFCC pysf'),
('kaggle_rnn_one', 99.25, 80.00, 3, 0.1, '9,13', '13 MFCC pysf')

update Models set model_type=0 where feature_description='Best Feature Set'
update Models set model_type=1 where feature_description='13 MFCC'
update Models set model_type=2 where feature_description='Mel Spectrograms'
update Models set model_type=3 where feature_description='13 MFCC pysf'

select * from Datasets
select * from Models