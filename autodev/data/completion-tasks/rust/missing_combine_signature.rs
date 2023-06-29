use std::{fmt::Debug, ops::Mul};

#[derive(Debug)]
struct Product<const ID: i64> {
    value: f64,
    description: String,
}

fn combine<todo> {
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
}
