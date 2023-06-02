class Greeter
    def initialize(names)
        @names = names
    end

    def say_hi
        if @names.respond_to?("join")
            puts <todo>
    end
end