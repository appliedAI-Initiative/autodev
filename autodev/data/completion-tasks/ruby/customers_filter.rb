Customer = Struct.new("Customer", :name, :gender, :age)

class CustomerCollection
  def initialize(customers)
    @customers = customers
  end

  def filter_male
    <todo>
  end
end

customers = []
customers << Customer.new("John", "m", 27)
customers << Customer.new("Mary", "f", 26)
customers << Customer.new("Anna", "f", 18)
customers << Customer.new("Bob", "m", 42)

puts CustomerCollection.new(customers).filter_male