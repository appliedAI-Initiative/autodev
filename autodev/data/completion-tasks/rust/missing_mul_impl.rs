use std::{fmt::Debug, ops::Mul};

#[derive(Debug)]
struct Product<const ID: i64> {
    value: f64,
    description: String,
}

// impl of Mul by 64 bit floats for Product
<todo>

fn combine<const ID: i64>(product_1: Product<ID>, product_2: Product<ID>) -> Product<ID> {
    Product::<ID> {
        value: product_1.value + product_2.value,
        description: format!("{}, {}", product_1.description, product_2.description)
    }
}

fn main() {
    let appliance_1 = Product::<987> {
        value: 9.0,
        description: "red appliance".to_string(),
    };

    let appliance_2 = Product::<987> {
        value: 12.0,
        description: "green appliance".to_string(),
    };

    let combined_appliance = combine(appliance_1, appliance_2);
    let Product { value: value_combined, description: description_combined } = &combined_appliance;

    println!("{value_combined} {description_combined}: {:?}", combined_appliance);

    let food = Product::<123> {
        value: 5.0,
        description: "food".to_string(),
    } * 0.75;

    // combined_and_another is a Vec containing both combined_appliance and food
    let combined_and_another: Vec<Box<dyn Debug>> = vec![Box::new(combined_appliance), Box::new(food)];

    for appliance in combined_and_another {
        println!("{:?}", appliance);
    }
}
