class math01():
    
    def __init__(self,x) -> None:
        self.x = x
        
    def cal01(self,y):
        self.z = self.x + y
        
    def cal02(self):
        ans = self.z*2
        return ans
ans = math01(1)
ans.cal01(2)
if __name__ == '__main__':
    print(ans.cal02())