def test_nonperiodic_dataset(nonperiodic_dataset):
    dataset, config = nonperiodic_dataset
    print(dataset.data)
    print(dataset.fixed_fields)


def test_periodic_dataset(periodic_dataset):
    dataset, config = periodic_dataset
    print(dataset.data)
    print(dataset.fixed_fields)
