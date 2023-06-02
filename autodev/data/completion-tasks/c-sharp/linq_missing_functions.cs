using System;
using System.Linq;
using System.Collections.Generic;


public class EmployeeAnalysis
{
	public static void Main()
	{
                // Example data
		IList<Employee> employees = new List<Employee>() {
			new Employee() { EmployeeID = 15, EmployeeName = "Alice", Age = 15 } ,
				new Employee() { EmployeeID = 12, EmployeeName = "Bob",  Age = 21 } ,
				new Employee() { EmployeeID = 33, EmployeeName = "Charlie",  Age = 18 } ,
				new Employee() { EmployeeID = 71, EmployeeName = "David" , Age = 20 } ,
				new Employee() { EmployeeID = 21, EmployeeName = "Emily" , Age = 21 }
		};

		var adultEmployeeNames = GetAdultEmployeeNames(employees); // Adults are over the age of 18 (inclusive).
		var oldestEmployee = GetOldestEmployee(employees);
	}

	<todo>
}

public class Employee{
	public int EmployeeID { get; set; }
	public string EmployeeName { get; set; }
	public int Age { get; set; }
}
