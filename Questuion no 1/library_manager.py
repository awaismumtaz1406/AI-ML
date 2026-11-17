
class Item:
    def __init__(self, title, item_id):
        self.title = title
        self.item_id = item_id

    def describe(self):
        return f"Item ID: {self.item_id}, Title: {self.title}"


class Book(Item):
    def __init__(self, title, item_id, author):
        super().__init__(title, item_id)
        self.author = author

    def describe(self):
        return f"Book: {self.title}, ID: {self.item_id}, Author: {self.author}"


class Magazine(Item):
    def __init__(self, title, item_id, issue_number):
        super().__init__(title, item_id)
        self.issue_number = issue_number

    def describe(self):
        return f"Magazine: {self.title}, ID: {self.item_id}, Issue: {self.issue_number}"


class ItemNotFoundError(Exception):
    pass


class Library:
    def __init__(self):
        self.items = {}

    def add_item(self, item):
        self.items[item.item_id] = item
        print(f"Added: {item.title}")

    def find_item(self, item_id):
        if item_id not in self.items:
            raise ItemNotFoundError(
                f"No item found with ID: {item_id}"
            )
        return self.items[item_id]

    def remove_item(self, item_id):
        item = self.find_item(item_id)
        del self.items[item_id]
        print(f"Removed: {item.title}")


if __name__ == "__main__":

    library = Library()

    # Adding books
    library.add_item(Book("Python Programming", "B101", "John Smith"))
    library.add_item(Book("Clean Code", "B102", "Robert Martin"))

    # Adding magazines
    library.add_item(Magazine("Science World", "M201", 15))
    library.add_item(Magazine("Tech Monthly", "M202", 28))

    # Display all items
    print("\n--- All Library Items ---")
    for item in library.items.values():
        print(item.describe())

    # Successful search
    print("\n--- Successful Search ---")
    try:
        item = library.find_item("B101")
        print("Found:", item.describe())
    except ItemNotFoundError as error:
        print("Exception:", error)

    # Missing ID search
    print("\n--- Missing-ID Search ---")
    try:
        library.find_item("X999")
    except ItemNotFoundError as error:
        print("Exception handled:", error)

    # Remove an item
    print("\n--- Remove an Item ---")
    library.remove_item("M201")

    # Display remaining items
    print("\n--- Items After Removal ---")
    for item in library.items.values():
        print(item.describe())
